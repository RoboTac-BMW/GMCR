import numpy as np
import open3d as o3d
from pathlib import Path
from numpy.random import default_rng
import json
from fitting_graphs.utility.graph import no_common_edge
import copy
import numpy.ma as ma
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from fitting_graphs.utility.preprocessing import remove_conflicting_correspondences
import time
from pypcd import pypcd


def load_pcd_color(path):
    pc = pypcd.PointCloud.from_path(path)

    x = pc.pc_data['x']
    y = pc.pc_data['y']
    z = pc.pc_data['z']
    points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
    if 'rgb' in pc.fields:
        colors = pc.pc_data["rgb"].view((np.uint8, 4))[:, 0:3] / 255.0
        colors = np.flip(colors, axis=1)
    else:
        colors = None

    return points, colors


class PointCloud:
    """
    Class wrapper for point clouds, which supports transformations and visualization with open3d.
    """
    def __init__(self, input, color=None, old_loader=False):

        if isinstance(input, str) or isinstance(input, Path):

            if old_loader:
                o3d_pcd = o3d.io.read_point_cloud(str(input))
                input = np.array(o3d_pcd.points)
            else:
                input, color = load_pcd_color(input)

        self.points = input
        self.shape = self.points.shape
        self.color = color
        self.keypoints = None
        self.descriptor = None

        if color is not None:
            assert self.color.shape[0] == self.points.shape[0]

    def __str__(self):

        return "Point Cloud with {} points.".format(self.points.shape[0])

    def __add__(self, other):

        if isinstance(other, PointCloud):
            self.points = self.points + other.points
            return self

        elif isinstance(other, np.ndarray):
            self.points = self.points + other
            return self

        else:
            raise NotImplementedError

    def __mul__(self, other):

        if isinstance(other, int) or isinstance(other, float) or (isinstance(other, np.ndarray) and np.isscalar(other)):
            self.points = other * self.points
            return self

        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def set_color(self, col):

        if col == 'blue':
            self.color = np.repeat([[0, 0, 1]], self.points.shape[0], axis=0)

        elif col == 'red':
            self.color = np.repeat([[1, 0, 0]], self.points.shape[0], axis=0)

    def voxel_downsample(self, voxel_size):

        self.points = np.array(self.to_o3d().voxel_down_sample(voxel_size).points)
        self.shape = self.points.shape

    def crop_bb(self, bb):

        max_bound = bb.max_bound + 0.02
        min_bound = bb.min_bound - 0.02

        points = self.points

        keep_inds = ((points > min_bound) * (points < max_bound)).all(axis=1)
        self.points = points[keep_inds]
        self.shape = self.points.shape

    def transform(self, R, t):
        """
        Applies a transformation given by R and t to the points in self.points
        :param R: SO(3) 3x3 rotation matrix
        :param t: 3x1 translation vector
        :return: None
        """
        assert R.shape == (3, 3)
        assert t.shape == (1, 3) or t.shape == (3,)

        if t.shape == (3,):
            t = t[None, :]

        self.points = (R@self.points.transpose()).transpose()
        self.points = self.points + t

    def transform_inverse(self, R, t, s=None):

        assert R.shape == (3, 3)
        assert t.shape == (1, 3)

        self.points = self.points - t
        if s is not None:
            R = 1/s * R
        self.points = (R.transpose() @ self.points.transpose()).transpose()

    def display(self):
        """
        Will display the point cloud using open3d. If colors are available they will be shown.
        :return:
        """
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.color is not None:
            print("> Showing colors")
            o3d_pcd.colors = o3d.utility.Vector3dVector(self.color)
        else:
            o3d_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(self.shape[0])])
        o3d.visualization.draw_geometries([o3d_pcd])

    def append(self, points):
        """
        Appends a set of points to the current set of points.
        :param points: numpy.ndarray with shape [num_points, 3]
        :return: None
        """
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 3

        self.points = np.concatenate([self.points, points], axis=0)
        self.shape = self.points.shape

    def to_o3d(self):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.color is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(self.color)
        return o3d_pcd


    def remove_ground_plane(self, distance_threshold, ransac_n, num_iterations):

        o3d_pcd = self.to_o3d()
        plane, inliers = o3d_pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        print(len(inliers))
        o3d_pcd = o3d_pcd.select_by_index(inliers, invert=True)
        self.points = np.array(o3d_pcd.points)
        self.color = np.array(o3d_pcd.colors)
        self.shape = self.points.shape


def get_gt_nn_corresp(source, target, max_dist=2):
    # nbrs_trg has shape [source, 1]

    nbrs_trg = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target)
    dist_trg, ind_trg = nbrs_trg.kneighbors(source, n_neighbors=1)

    corresp = np.concatenate([np.arange(source.shape[0])[:, None], ind_trg[:, 0][:, None]], axis=1)
    not_clip = (dist_trg[:, 0] < max_dist)
    corresp = corresp[not_clip]
    # corresp, _ = remove_conflicting_correspondences(corresp)

    """
    # nbrs_source has shape [target, 1]
    nbrs_src = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source)
    dist_src, ind_src = nbrs_src.kneighbors(target, n_neighbors=1)

    consistent_src_bool = (ind_trg[ind_src[:, 0], 0] == ind_src[:, 0])
    consistent_trg_bool = (ind_src[ind_trg[:, 0], 0] == ind_trg[:, 0])
    ind_src = ind_src[consistent_src_bool, 0]
    ind_trg = ind_trg[consistent_trg_bool, 0]

    dist_src = dist_src[consistent_src_bool]
    dist_trg = dist_trg[consistent_trg_bool]

    assert np.allclose(dist_src, dist_trg)

    not_clip = (dist_src <= max_dist)
    ind_src = ind_src[not_clip]
    ind_trg = ind_trg[not_clip]



    return np.concatenate([ind_src[:, None], ind_trg[:, None]], axis=1)
    """
    return corresp


class PointCloudPair:
    """
    Class wrapper for a pair of point clouds, which are related by a known transformation
    """
    def __init__(self, source, target, R, t, s, C_true=None, C_false=None, c_putative=None):

        assert isinstance(source, PointCloud)
        assert isinstance(target, PointCloud)

        self.source = source
        self.org_source = copy.deepcopy(source)
        self.org_target = copy.deepcopy(target)
        self.target = target
        self.R = R
        self.t = t
        self.s = s
        self.C_true = C_true
        self.C_false = C_false
        self.c_putative = c_putative

    def __str__(self):

        return "PointCloudPair with: \n\t{} source points\n\t{} target points \n\t{} true correspondences\n\t" \
               "{} false correspondences".format(self.source.points.shape[0], self.target.points.shape[0],
                                                 self.C_true.shape[0], self.C_false.shape[0])

    def str_gt_transform(self):

        return "\n ---------- GT Transformation ----------\n Scale: {} \n " \
               "Rotation: {} \n Translation:{}".format(self.s, self.R, self.t)

    def transform_source(self, R, t):
        """
        Applies a homogeneous transformation to the source point cloud
        :param R: np.array(3x3) with an SO(3) rotation matrix
        :param t: np.array(1x3) with the translation vector
        :return: None
        """

        self.source.transform(R, t)

    def transform_target(self, R, t):

        self.target.transform(R, t)

    def transform_target_inverse(self, R, t, s=None):

        self.target.transform_inverse(R, t, s)

    def transform_source_inverse(self, R, t):

        self.source.transform_inverse(R, t)

    def sample_correspondences(self, outlier_ratio, num_correspondences, local=False, inlier_nns=4, random_points=True):
        """
        Samples a subset of inlier correspondences self.C_true and a subset of the outlier correspondences C_False
        :param outlier_ratio: float for ratio outilier / num_correspondences
        :param num_correspondences: total number of correspondences returned
        :return: tuple(inlier_subset, outlier_subset) as np.ndarray
        """

        #assert self.C_true is not None and self.C_false is not None
        random_generator = default_rng()
        num_outliers = round(num_correspondences * outlier_ratio)
        num_inliers = num_correspondences - num_outliers
        assert num_inliers > 0

        # First sample inliers
        inlier_inds = random_generator.choice(self.C_true.shape[0], size=num_inliers, replace=False)
        inlier_c = self.C_true[inlier_inds]

        if local:
            # Sample subset of correspondences according to outlier ratio

            # Take points next to actual point in the source point cloud
            inlier_points = self.source.points[inlier_c[:, 0]]
            nbrs = NearestNeighbors(n_neighbors=inlier_nns, algorithm='ball_tree').fit(self.source.points)
            dist, ind = nbrs.kneighbors(inlier_points, n_neighbors=inlier_nns)
            # ToDo: Sample among the NNs and not just take the furthest one
            #inlier_c[:, 0] = ind[:, -1]
            rand_nns = random_generator.choice(ind.shape[1], size=ind.shape[0], replace=True)
            inlier_c[:, 0] = ind[np.arange(ind.shape[0]), rand_nns]

            # Take points next to actual point in the target point cloud
            inlier_points = self.target.points[inlier_c[:, 1]]
            nbrs = NearestNeighbors(n_neighbors=inlier_nns, algorithm='ball_tree').fit(self.target.points)
            dist, ind = nbrs.kneighbors(inlier_points, n_neighbors=inlier_nns)
            # ToDo: Sample among the NNs and not just take the furthest one
            #inlier_c[:, 1] = ind[:, -1]
            rand_nns = random_generator.choice(ind.shape[1], size=ind.shape[0], replace=True)
            inlier_c[:, 1] = ind[np.arange(ind.shape[0]), rand_nns]
            inlier_c, _ = remove_conflicting_correspondences(inlier_c)
            #a = np.unique(ind[:, -3])
            #print(a.shape)
            #self.c_putative = copy.copy(inlier_c)
            #self.c_putative[:, 1] = ind[:, -1]

            # Sample outliers
            if num_outliers > 0:
                outlier_c = self.sample_false_correspondences(num_outliers, inlier_c, local, outlier_ratio, random_points=random_points)
            else:
                outlier_c = np.empty([0, 2], dtype=np.int32)

            #assert no_common_edge(inlier_c, outlier_c)

            #inlier_noise = np.random.uniform(-0.07, 0.07, (inlier_c.shape[0], 3))
            #self.target.points[inlier_c[:, 1]] = self.org_target.points[inlier_c[:, 1]] + inlier_noise
            #inlier_noise = np.random.uniform(-0.04, 0.04, (inlier_c.shape[0], 3))
            #self.source.points[inlier_c[:, 0]] = self.org_source.points[inlier_c[:, 0]] + inlier_noise
            return inlier_c, outlier_c

        elif not local:
            inlier_points = self.source.points[inlier_c[:, 0]]

            # Sample outliers
            if num_outliers > 0:
                outlier_c = self.sample_false_correspondences(num_outliers, inlier_c, local, outlier_ratio, random_points=random_points)

            else:
                outlier_c = np.empty([0, 2], dtype=np.int32)

            if not local:
                assert no_common_edge(inlier_c, outlier_c)

            #inlier_noise = np.random.uniform(-0.04, 0.04, (inlier_c.shape[0], 3))
            #self.target.points[inlier_c[:, 1]] = self.org_target.points[inlier_c[:, 1]] + inlier_noise
            #inlier_noise = np.random.uniform(-0.04, 0.04, (inlier_c.shape[0], 3))
            #self.source.points[inlier_c[:, 0]] = self.org_source.points[inlier_c[:, 0]] + inlier_noise
            return inlier_c, outlier_c


    def sample_false_correspondences(self, num_outliers, given_inliers=None, local=False, o_r=None, random_points=True):

        num_points = self.source.shape[0]
        num_outlier_bases = np.ceil(o_r * 10).astype(int) + 2
        false_points = self.target.shape[0] - num_points
        random_generator = np.random.default_rng()

        # assert self.source.shape == self.target.shape

        if local:

            # Sample base outliers
            possible_start_points = np.arange(num_points)
            if given_inliers is not None:
                to_remove = np.isin(possible_start_points, given_inliers[:, 0])
                possible_start_points = possible_start_points[np.invert(to_remove)]

            outlier_start = random_generator.choice(possible_start_points, size=num_outlier_bases, replace=False)
            possible_end_inds = np.repeat(np.arange(num_points)[None, :], num_outlier_bases, axis=0)  # inds per start point
            inds_to_keep = np.invert(possible_end_inds == outlier_start[:, None])  # remove the inds corresp. to start
            possible_end_inds = possible_end_inds[inds_to_keep].reshape(num_outlier_bases, -1)  # is flattened, so reshape to matrix
            if given_inliers is not None:
                to_remove = np.isin(possible_end_inds, given_inliers[:, 1], invert=False)
                possible_end_inds = [np.delete(possible_end_inds[i, :], to_remove[i, :]) for i in range(to_remove.shape[0])]

            randinds = [np.random.randint(0, possible_end_inds[i].shape[0]) for i in range(len(possible_end_inds))]
            outlier_end = np.array([possible_end_inds[i][randinds[i]] for i in range(len(possible_end_inds))])

            base_outlier_c = np.concatenate([outlier_start[:, None], outlier_end[:, None]], axis=1)

            # Sample more outlier points next to base outliers
            def dist_int_over_bins(k, num_b, max_val):

                bins = np.zeros(num_b)
                for i in range(k):
                    rand_ind = np.random.randint(num_b)
                    while bins[rand_ind] + 1 > max_val:
                        rand_ind = np.random.randint(num_b)
                    bins[rand_ind] += 1
                return bins.astype(int)

            s_num_neigh = 10
            t_num_neigh = 20
            outlier_dist = dist_int_over_bins(num_outliers, base_outlier_c.shape[0], s_num_neigh - 1)
            source_outlier_points = self.source.points[base_outlier_c[:, 0]]
            nbrs = NearestNeighbors(n_neighbors=s_num_neigh, algorithm='ball_tree').fit(self.source.points)
            dist, ind = nbrs.kneighbors(source_outlier_points, n_neighbors=s_num_neigh)
            rand_inds = [random_generator.choice(np.arange(1, s_num_neigh), size=i, replace=False) for i in outlier_dist]
            start_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
            start_inds = np.concatenate(start_inds)

            # possible_start_inds = ind[:, 1:].flatten()
            # rand_inds = random_generator.choice(num_neigh, size=num_outliers, replace=False)

            target_inlier_points = self.target.points[base_outlier_c[:, 1]]
            nbrs = NearestNeighbors(n_neighbors=t_num_neigh, algorithm='ball_tree').fit(self.target.points)
            dist, ind = nbrs.kneighbors(target_inlier_points, n_neighbors=t_num_neigh)
            rand_inds = [random_generator.choice(np.arange(1, t_num_neigh), size=i, replace=False) for i in outlier_dist]
            end_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
            #ind = np.array(ind)
            #end_inds = [ind[i, -d:] for i, d in enumerate(outlier_dist) if d > 0]
            end_inds = np.concatenate(end_inds)

            outlier_c = np.concatenate([start_inds[:, None], end_inds[:, None]], axis=1)

        else:

            # Sample start points of outliers
            possible_start_points = np.arange(num_points)
            if given_inliers is not None:

                # Remove inlier starts as possible outlier starts
                to_remove = np.isin(possible_start_points, given_inliers[:, 0])
                possible_start_points = possible_start_points[np.invert(to_remove)]

            # Sample outlier start points
            outlier_start = random_generator.choice(possible_start_points, size=num_outliers, replace=False)

            # Sample end points, while outlier_start != outlier_end
            #possible_end_inds = np.arange(num_points)
            #if given_inliers is not None:
            #    to_remove = np.isin(possible_end_inds, given_inliers[:, 1])
            #    possible_end_inds = possible_end_inds[np.invert(to_remove)]

            possible_end_inds = np.repeat(np.arange(num_points)[None, :], num_outliers, axis=0)  # inds per start point
            inds_to_keep = np.invert(possible_end_inds == outlier_start[:, None])  # remove the inds corresp. to start

            possible_end_inds = possible_end_inds[inds_to_keep].reshape(num_outliers, -1)  # is flattened, so reshape to matrix

            if given_inliers is not None:

                to_remove = np.isin(possible_end_inds, given_inliers[:, 1], invert=False)
                possible_end_inds = [np.delete(possible_end_inds[i, :], to_remove[i, :]) for i in range(to_remove.shape[0])]

            # Add the outlier points to the possible inds
            if random_points:
                false_inds = np.arange(num_points, num_points + false_points)
                possible_end_inds = [np.concatenate([possible_end_inds[i], false_inds], axis=0) for i in range(len(possible_end_inds))]

            randinds = [np.random.randint(0, possible_end_inds[i].shape[0]) for i in range(len(possible_end_inds))]
            outlier_end = np.array([possible_end_inds[i][randinds[i]] for i in range(len(possible_end_inds))])

            outlier_c = np.concatenate([outlier_start[:, None], outlier_end[:, None]], axis=1)


        return outlier_c

    """
        if not local:
            # Sample start points of outliers
            possible_start_points = np.arange(num_points)
            if given_inliers is not None:

                # Remove inlier starts as possible outlier starts
                to_remove = np.isin(possible_start_points, given_inliers[:, 0])
                possible_start_points = possible_start_points[np.invert(to_remove)]

            # Sample outlier start points
            outlier_start = random_generator.choice(possible_start_points, size=num_outliers, replace=False)

            # Sample end points, while outlier_start != outlier_end
            #possible_end_inds = np.arange(num_points)
            #if given_inliers is not None:
            #    to_remove = np.isin(possible_end_inds, given_inliers[:, 1])
            #    possible_end_inds = possible_end_inds[np.invert(to_remove)]

            possible_end_inds = np.repeat(np.arange(num_points)[None, :], num_outliers, axis=0)  # inds per start point
            inds_to_keep = np.invert(possible_end_inds == outlier_start[:, None])  # remove the inds corresp. to start

            possible_end_inds = possible_end_inds[inds_to_keep].reshape(num_outliers, -1)  # is flattened, so reshape to matrix

            if given_inliers is not None:

                to_remove = np.isin(possible_end_inds, given_inliers[:, 1], invert=False)
                possible_end_inds = [np.delete(possible_end_inds[i, :], to_remove[i, :]) for i in range(to_remove.shape[0])]

            # Add the outlier points to the possible inds
            false_inds = np.arange(num_points, num_points + false_points)
            possible_end_inds = [np.concatenate([possible_end_inds[i], false_inds], axis=0) for i in range(len(possible_end_inds))]

            randinds = [np.random.randint(0, possible_end_inds[i].shape[0]) for i in range(len(possible_end_inds))]
            outlier_end = np.array([possible_end_inds[i][randinds[i]] for i in range(len(possible_end_inds))])

            outlier_c = np.concatenate([outlier_start[:, None], outlier_end[:, None]], axis=1)

        elif local:

            def dist_int_over_bins(k, num_b, max_val):

                bins = np.zeros(num_b)
                for i in range(k):
                    rand_ind = np.random.randint(num_b)
                    while bins[rand_ind] + 1 > max_val:
                        rand_ind = np.random.randint(num_b)
                    bins[rand_ind] += 1
                return bins.astype(int)

            num_neigh = 10
            
            
            rand_outlier_dist = random_generator.choice(given_inliers.shape[0], size=num_outliers, replace=True)
            _, u_ind, u_counts = np.unique(rand_outlier_dist, return_counts=True, return_index=True)

            outlier_dist = np.zeros(given_inliers.shape[0])
            outlier_dist[u_ind] = u_counts
            outlier_dist = outlier_dist.astype(int)
            
            num_outliers_assoc = round(num_outliers / 2)
            num_outliers_gross = num_outliers - num_outliers_assoc
            outlier_dist = dist_int_over_bins(num_outliers_assoc, given_inliers.shape[0], num_neigh - 1)
            source_inlier_points = self.source.points[given_inliers[:, 0]]
            nbrs = NearestNeighbors(n_neighbors=num_neigh, algorithm='ball_tree').fit(self.source.points)
            dist, ind = nbrs.kneighbors(source_inlier_points, n_neighbors=num_neigh)
            rand_inds = [random_generator.choice(np.arange(1, num_neigh), size=i, replace=False) for i in outlier_dist]
            start_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
            start_inds = np.concatenate(start_inds)

            #possible_start_inds = ind[:, 1:].flatten()
            #rand_inds = random_generator.choice(num_neigh, size=num_outliers, replace=False)

            target_inlier_points = self.target.points[given_inliers[:, 1]]
            nbrs = NearestNeighbors(n_neighbors=num_neigh, algorithm='ball_tree').fit(self.target.points)
            dist, ind = nbrs.kneighbors(target_inlier_points, n_neighbors=num_neigh)
            rand_inds = [random_generator.choice(np.arange(1, num_neigh), size=i, replace=False) for i in outlier_dist]
            end_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
            end_inds = np.concatenate(end_inds)

            outlier_c = np.concatenate([start_inds[:, None], end_inds[:, None]], axis=1)

            possible_start_points = np.arange(num_points)
            if given_inliers is not None:

                # Remove inlier starts as possible outlier starts
                to_remove = np.isin(possible_start_points, given_inliers[:, 0])
                possible_start_points = possible_start_points[np.invert(to_remove)]

            # Sample outlier start points
            outlier_start = random_generator.choice(possible_start_points, size=num_outliers_gross, replace=False)


            possible_end_inds = np.repeat(np.arange(num_points)[None, :], num_outliers_gross, axis=0)  # inds per start point
            inds_to_keep = np.invert(possible_end_inds == outlier_start[:, None])  # remove the inds corresp. to start

            possible_end_inds = possible_end_inds[inds_to_keep].reshape(num_outliers_gross, -1)  # is flattened, so reshape to matrix

            if given_inliers is not None:

                to_remove = np.isin(possible_end_inds, given_inliers[:, 1], invert=False)
                possible_end_inds = [np.delete(possible_end_inds[i, :], to_remove[i, :]) for i in range(to_remove.shape[0])]

            # Add the outlier points to the possible inds
            false_inds = np.arange(num_points, num_points + false_points)
            possible_end_inds = [np.concatenate([possible_end_inds[i], false_inds], axis=0) for i in range(len(possible_end_inds))]

            randinds = [np.random.randint(0, possible_end_inds[i].shape[0]) for i in range(len(possible_end_inds))]
            outlier_end = np.array([possible_end_inds[i][randinds[i]] for i in range(len(possible_end_inds))])

            outlier_c = np.concatenate([outlier_c, np.concatenate([outlier_start[:, None], outlier_end[:, None]], axis=1)], axis=0)

        return outlier_c
    """
    def display(self, subsample=False, viewer_file=None, show_corresp=True, c_inlier=None, c_outlier=None):
        """
        Displays the point cloud pair, the inlier correspondences if present and the outlier correspondences if present.
        :param subsample: If true the correspondences are downsampled before visualization
        :return: None
        """
        vis_lst = []

        o3d_pcd_source = o3d.geometry.PointCloud()
        o3d_pcd_target = o3d.geometry.PointCloud()
        source_points = self.source.points
        target_points = self.target.points
        if subsample:
            target_points = target_points[:target_points.shape[0] - 200]
            source_points = self.source.points[np.random.choice(np.arange(source_points.shape[0]), size=600)]
            #target_points = self.target.points[np.random.choice(np.arange(target_points.shape[0]), size=800)]

        o3d_pcd_source.points = o3d.utility.Vector3dVector(source_points)
        o3d_pcd_target.points = o3d.utility.Vector3dVector(target_points)

        if self.source.color is None:
            o3d_pcd_source.paint_uniform_color(np.array([0, 0, 1]))
        else:
            o3d_pcd_source.colors = o3d.utility.Vector3dVector(self.source.color)

        if self.target.color is None:
            o3d_pcd_target.paint_uniform_color(np.array([1, 0, 0]))
        else:
            o3d_pcd_target.colors = o3d.utility.Vector3dVector(self.target.color)

        print("Displaying:\nSource->Blue\nTarget->Purple")
        mat1 = o3d.visualization.rendering.MaterialRecord()
        mat1.shader = 'defaultUnlit'
        mat1.line_width = 50
        mat1.point_size = 20

        vis_lst.append({'name': 'src_pcd', 'geometry': o3d_pcd_source, 'material': mat1})
        vis_lst.append({'name': 'trg_pcd', 'geometry': o3d_pcd_target, 'material': mat1})

        if show_corresp:
            # ToDo: Change this back
            if c_inlier is None:
                c_inlier, c_outlier = self.sample_correspondences(0.2, 5, local=False)
            line_points = np.concatenate([self.source.points, self.target.points], axis=0)
            line_ids = copy.deepcopy(c_inlier)
            line_ids[:, 1] += self.source.points.shape[0]

            true_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_ids),
            )
            colors = [[0, 1, 0] for i in range(len(line_ids))]
            true_line_set.colors = o3d.utility.Vector3dVector(colors)
            mat2 = o3d.visualization.rendering.MaterialRecord()
            mat2.shader = 'unlitLine'
            mat2.line_width = 20
            mat2.point_size = 12
            vis_lst.append({'name': 'inlier_c', 'geometry': true_line_set, 'material': mat2})

            outlier_line_ids = copy.deepcopy(c_outlier)
            outlier_line_ids[:, 1] += self.source.points.shape[0]

            false_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(outlier_line_ids),
            )
            colors = [[1, 0, 0] for i in range(len(outlier_line_ids))]
            false_line_set.colors = o3d.utility.Vector3dVector(colors)
            vis_lst.append({'name': 'outlier_c', 'geometry': false_line_set, 'material': mat2})
            print("C_True->Red")

        """
        if self.C_true is not None:
            line_points = np.concatenate([self.source.points, self.target.points], axis=0)
            line_ids = self.C_true.copy()
            line_ids[:, 1] += self.source.points.shape[0]

            # subsample correspondences
            if subsample:
                random_inds = np.random.choice(np.arange(line_ids.shape[0]), 50, replace=False)
                line_ids = line_ids[random_inds, :]

            true_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_ids),
            )
            colors = [[0, 1, 0] for i in range(len(line_ids))]
            true_line_set.colors = o3d.utility.Vector3dVector(colors)
            vis_lst.append(true_line_set)
            print("C_True->Green")
        """

        """
        if self.C_false is not None:

            if subsample:
                outlier_c = self.sample_false_correspondences(100)
            else:
                outlier_c = self.sample_false_correspondences(1000)

            line_points = np.concatenate([self.source.points, self.target.points], axis=0)
            outlier_c[:, 1] += self.source.points.shape[0]

            false_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(outlier_c),
            )
            colors = [[1, 0, 0] for i in range(len(outlier_c))]
            false_line_set.colors = o3d.utility.Vector3dVector(colors)
            vis_lst.append(false_line_set)
            print("C_True->Red")
        """

        if viewer_file is None:
        #    vis_lst.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
            o3d.visualization.gui.Application.instance.initialize()
            w = o3d.visualization.O3DVisualizer('Visualizer', 1960, 1080)
            w.show_skybox(False)
            o3d.visualization.gui.Application.instance.add_window(w)
            [w.add_geometry(geo) for geo in vis_lst]
            ctr = o3d.visualization.ViewControl()
            print(ctr.convert_to_pinhole_camera_parameters())
            o3d.visualization.gui.Application.instance.run()


        """
            width, height = get_window_dimensions_from_file(file_path=viewer_file)
            o3d.visualization.gui.Application.instance.initialize()
            w = o3d.visualization.O3DVisualizer('Visualizer', width=width, height=height)
            w.show_skybox(False)
            o3d.visualization.gui.Application.instance.add_window(w)
            #vis.create_window(width=width, height=height)
            ctr = o3d.visualization.ViewControl()
            param = o3d.io.read_pinhole_camera_parameters(viewer_file)
            [w.add_geometry(geo) for geo in vis_lst]
            ctr.convert_from_pinhole_camera_parameters(param)
            param = o3d.io.read_pinhole_camera_parameters(viewer_file)
            intrinsic = param.intrinsic.intrinsic_matrix
            extrinsic = param.extrinsic
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            param.intrinsic.intrinsic_matrix = intrinsic
            param.extrinsic = extrinsic
            w.setup_camera(param.intrinsic, param.extrinsic)
            print(param)
            print(ctr.convert_to_pinhole_camera_parameters())
            time.sleep(2)
            o3d.visualization.gui.Application.instance.run()
            #vis.destroy_window()
        """
        o3d.visualization.draw_geometries([v['geometry'] for v in vis_lst])
    def display_putative(self, no_c=False):
        vis_lst = []
        o3d_pcd_source = o3d.geometry.PointCloud()
        o3d_pcd_target = o3d.geometry.PointCloud()
        o3d_pcd_source.points = o3d.utility.Vector3dVector(self.source.points)
        o3d_pcd_target.points = o3d.utility.Vector3dVector(self.target.points)

        #if self.source.color is None:
        o3d_pcd_source.paint_uniform_color(np.array([0, 0, 1]))
        #else:
        #    o3d_pcd_source.colors = o3d.utility.Vector3dVector(self.source.color)

        if self.target.color is None:
            o3d_pcd_target.paint_uniform_color(np.array([1, 0, 1]))
        else:
            o3d_pcd_target.colors = o3d.utility.Vector3dVector(self.target.color)

        vis_lst.append(o3d_pcd_source)
        vis_lst.append(o3d_pcd_target)

        if self.c_putative is not None and no_c is False:
            line_points = np.concatenate([self.source.points, self.target.points], axis=0)
            line_ids = self.c_putative.copy()
            line_ids[:, 1] += self.source.points.shape[0]

            true_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_ids),
            )
            colors = [[1, 0.6, 0] for i in range(len(line_ids))]
            true_line_set.colors = o3d.utility.Vector3dVector(colors)
            vis_lst.append(true_line_set)
            print("C_putative->Green")

        vis_lst.reverse()
        o3d.visualization.draw_geometries(vis_lst)

    def find_correspondences_nn(self, num_c, num_outlier_c=None, local_outlier=False):

        NN_ind = 4
        dist_clip = 0.2
        random_generator = default_rng()
        inlier_nns = 5

        t_source = copy.deepcopy(self.source)
        t_source = t_source * self.s
        t_source.transform(self.R, self.t)

        inlier_c = get_gt_nn_corresp(t_source.points, self.target.points, dist_clip)
        # up to here it looks fine

        rand_inliers = random_generator.choice(inlier_c.shape[0], size=num_c, replace=False)
        inlier_c = inlier_c[rand_inliers, :]


        inlier_points = self.source.points[inlier_c[:, 0]]
        nbrs = NearestNeighbors(n_neighbors=inlier_nns, algorithm='ball_tree').fit(self.source.points)
        dist, ind = nbrs.kneighbors(inlier_points, n_neighbors=inlier_nns)
        # ToDo: Sample among the NNs and not just take the furthest one
        #inlier_c[:, 0] = ind[:, -1]
        rand_nns = random_generator.choice(ind.shape[1], size=ind.shape[0], replace=True)
        inlier_c[:, 0] = ind[np.arange(ind.shape[0]), rand_nns]

        # Take points next to actual point in the target point cloud
        inlier_points = self.target.points[inlier_c[:, 1]]
        nbrs = NearestNeighbors(n_neighbors=inlier_nns, algorithm='ball_tree').fit(self.target.points)
        dist, ind = nbrs.kneighbors(inlier_points, n_neighbors=inlier_nns)
        # ToDo: Sample among the NNs and not just take the furthest one
        #inlier_c[:, 1] = ind[:, -1]
        rand_nns = random_generator.choice(ind.shape[1], size=ind.shape[0], replace=True)
        inlier_c[:, 1] = ind[np.arange(ind.shape[0]), rand_nns]
        corresp, _ = remove_conflicting_correspondences(inlier_c)
        self.C_true = corresp
        if num_outlier_c is not None:

            if not local_outlier:
                rand_start = random_generator.choice(self.source.shape[0], size=num_outlier_c, replace=False)
                rand_stop = random_generator.choice(self.target.shape[0], size=num_outlier_c, replace=False)
                outlier_c = np.concatenate([rand_start[:, None], rand_stop[:, None]], axis=1)
                self.C_false = outlier_c
                corresp = np.concatenate([corresp, outlier_c], axis=0)

            else:

                num_outlier_bases = 10
                n_src_points = self.source.shape[0]
                n_trg_points = self.target.shape[0]
                # Sample base outliers
                possible_start_points = np.arange(n_src_points)
                possible_end_inds = np.arange(n_trg_points)

                outlier_start = random_generator.choice(possible_start_points, size=num_outlier_bases, replace=False)
                outlier_end = random_generator.choice(possible_end_inds, size=num_outlier_bases, replace=False)

                base_outlier_c = np.concatenate([outlier_start[:, None], outlier_end[:, None]], axis=1)

                # Sample more outlier points next to base outliers
                def dist_int_over_bins(k, num_b, max_val):

                    bins = np.zeros(num_b)
                    for i in range(k):
                        rand_ind = np.random.randint(num_b)
                        while bins[rand_ind] + 1 > max_val:
                            rand_ind = np.random.randint(num_b)
                        bins[rand_ind] += 1
                    return bins.astype(int)

                s_num_neigh = 10
                #t_num_neigh = 20
                t_num_neigh = 20
                outlier_dist = dist_int_over_bins(num_outlier_c, base_outlier_c.shape[0], s_num_neigh - 1)
                source_outlier_points = self.source.points[base_outlier_c[:, 0]]
                nbrs = NearestNeighbors(n_neighbors=s_num_neigh, algorithm='ball_tree').fit(self.source.points)
                dist, ind = nbrs.kneighbors(source_outlier_points, n_neighbors=s_num_neigh)
                rand_inds = [random_generator.choice(np.arange(1, s_num_neigh), size=i, replace=False) for i in
                             outlier_dist]
                start_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
                start_inds = np.concatenate(start_inds)

                # possible_start_inds = ind[:, 1:].flatten()
                # rand_inds = random_generator.choice(num_neigh, size=num_outliers, replace=False)

                target_inlier_points = self.target.points[base_outlier_c[:, 1]]
                nbrs = NearestNeighbors(n_neighbors=t_num_neigh, algorithm='ball_tree').fit(self.target.points)
                dist, ind = nbrs.kneighbors(target_inlier_points, n_neighbors=t_num_neigh)
                rand_inds = [random_generator.choice(np.arange(1, t_num_neigh), size=i, replace=False) for i in
                             outlier_dist]
                end_inds = [ind[i, rand_inds[i]] for i in range(len(rand_inds))]
                # ind = np.array(ind)
                # end_inds = [ind[i, -d:] for i, d in enumerate(outlier_dist) if d > 0]
                end_inds = np.concatenate(end_inds)

                outlier_c = np.concatenate([start_inds[:, None], end_inds[:, None]], axis=1)
                corresp = np.concatenate([corresp, outlier_c], axis=0)
                self.C_false = outlier_c

        corresp, _ = remove_conflicting_correspondences(corresp)
        print(f"> Found {self.C_true.shape[0]} inliers | {self.C_false.shape[0]} outliers")
        self.c_putative = corresp


    def find_correspondences_fpfh(self, voxel_size, num_corresp=None):

        def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
            feat1tree = cKDTree(feat1)
            dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
            if return_distance:
                return nn_inds, dists
            else:
                return nn_inds

        def find_correspondences_nn(feats0, feats1, mutual_filter=True):
            nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
            corres01_idx0 = np.arange(len(nns01))
            corres01_idx1 = nns01

            if not mutual_filter:
                return corres01_idx0, corres01_idx1

            nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
            corres10_idx1 = np.arange(len(nns10))
            corres10_idx0 = nns10

            mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
            corres_idx0 = corres01_idx0[mutual_filter]
            corres_idx1 = corres01_idx1[mutual_filter]

            return corres_idx0, corres_idx1

        def find_corresp_cos(feats0, feats1, num):

            feats0_norm = feats0 / np.linalg.norm(feats0, axis=1, keepdims=True)
            feats1_norm = feats1 / np.linalg.norm(feats1, axis=1, keepdims=True)

            cos_sim = feats0_norm @ feats1_norm.T
            cos_sim = np.nan_to_num(cos_sim, nan=0)
            corrsp_src = np.arange(feats0.shape[0])
            corrsp_trg = np.argmax(cos_sim, axis=1)

            return corrsp_src, corrsp_trg

        radius_normal = 2 * voxel_size
        radius_feature = voxel_size * 2

      #  o3d_source_pcd = self.source.to_o3d()
        o3d_source_pcd = self.source.to_o3d().voxel_down_sample(voxel_size)
        self.source.points = np.array(o3d_source_pcd.points)
        o3d_source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        #o3d_target_pcd = self.target.to_o3d()
        o3d_target_pcd = self.target.to_o3d().voxel_down_sample(voxel_size)
        self.target.points = np.array(o3d_target_pcd.points)
        o3d_target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        source_features = np.array(o3d.pipelines.registration.compute_fpfh_feature(o3d_source_pcd,
                                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                                       radius=radius_feature,
                                                                                       max_nn=100)).data).T
        target_features = np.array(o3d.pipelines.registration.compute_fpfh_feature(o3d_target_pcd,
                                                                                  o3d.geometry.KDTreeSearchParamHybrid(
                                                                                      radius=radius_feature,
                                                                                      max_nn=100)).data).T
        #print(target_features.shape)
        #print("Hallo")
        corrs_source, corrs_target = find_correspondences_nn(source_features, target_features, mutual_filter=True)
        #corrs_source, corrs_target = find_corresp_cos(source_features, target_features, num_corresp)

        if num_corresp is not None and corrs_source.shape[0] > num_corresp:
            random_generator = default_rng()
            rand_inds = random_generator.choice(corrs_source.shape[0], size=num_corresp, replace=False)
            corrs_source = corrs_source[rand_inds]
            corrs_target = corrs_target[rand_inds]

        # Create putative correspondences
        putative_corresp = np.concatenate([corrs_source[:, None],
                                           corrs_target[:, None]], axis=1)

        self.c_putative = putative_corresp

def get_window_dimensions_from_file(file_path):

    with open(file_path, 'r') as f:

        content = json.load(f)

    width, height = content['intrinsic']['width'], content['intrinsic']['height']

    return width, height