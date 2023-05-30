import open3d as o3d
import numpy as np
from fitting_graphs.utility.registration import RegistrationResult
from fitting_graphs.base_classes import RegistrationMethod

# For the sake of readability
ransac_corresp = o3d.pipelines.registration.registration_ransac_based_on_correspondence


class RANSACWrapper(RegistrationMethod):

    name = "RANSAC"

    def __init__(self, parameters):
        super(RANSACWrapper, self).__init__(parameters)
        self.iterations = self.parameters['iterations']
        self.max_corresp_pair_dist = self.parameters['max_corresp_pair_dist']
        self.ransac_n = self.parameters['ransac_n']
        self.confidence = self.parameters['confidence']

    def __call__(self, source_pcd, target_pcd, correspondences, variables=None, inlier_inds=None):

        # setup the ransac parameters
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        if 's' in variables:
            estimation_method.with_scaling = True
        else:
            estimation_method.with_scaling = False

        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria()
        criteria.max_iteration = self.iterations
        criteria.confidence = self.confidence

        # Convert point clouds to open3d
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pcd.points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pcd.points)
        corresp = o3d.utility.Vector2iVector(correspondences)

        # run registration
        res = ransac_corresp(source=source, target=target, corres=corresp,
                             max_correspondence_distance=self.max_corresp_pair_dist, estimation_method=estimation_method,
                             ransac_n=self.ransac_n, checkers=[], criteria=criteria)

        # return result
        T = res.transformation
        R_s = T[0:3, 0:3]
        if 's' in variables:
            s = np.sqrt((R_s @ R_s.transpose())[0, 0])
            print(s)
        else:
            s = 1
        R = T[0:3, 0:3] / s
        t = T[:3, 3][None, :]

        return RegistrationResult(s, R, t), None


 