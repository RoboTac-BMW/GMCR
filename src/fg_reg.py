import sys
sys.path.append('/home/mgentner/repos/MT_CertifiableRobustPCR')

import numpy as np
from fitting_graphs.utility.graph import Graph
from fitting_graphs.datasets.stanford_scanning_repository import StanfordDataset
from fitting_graphs.base_classes import RegistrationMethod
from fitting_graphs.utility.registration import RegistrationResult
from fitting_graphs.utility.preprocessing import remove_conflicting_correspondences, mc_inlier_scale, mc_inlier_rotation
from fitting_graphs.src.invariants import compute_tims, compute_trims, compute_scale_estimate, compute_rotation_estimate, compute_translation_estimate
from fitting_graphs.src.consensus_functions import con_func_scale_3d, cp_con_func_rotation_3d, con_func_translation_3d, con_func_rotation_3d
import time
from fitting_graphs.utility.timer import Timer


class FittingGraphRegistration(RegistrationMethod):

    name = 'FG_REG'

    def __init__(self, parameters):
        super(FittingGraphRegistration, self).__init__(parameters)

    def __call__(self, source, target, correspondences, variables=None, inlier_mask=None):
        """

        :param source: PointCloud of shape [num_s_points, 3]
        :param target: PointCloud of shape [num_t_points, 3]
        :param correspondences: np.array() of shape [num_correspondences, 2]
        :param variables: list of str from ['s', 'R', 't']
        :return: RegistrationResult
        """

        reg_res = RegistrationResult()
        reg_res.true_inlier_mask = inlier_mask

        c = self.parameters['threshold']
        beta_i = self.parameters['beta_i']

        # Removing correspondences, that start or end at the same point
        clean_correspondences, keep_boolean_mask = remove_conflicting_correspondences(correspondences)
        if inlier_mask is not None:
            kept_inlier_mask = inlier_mask[keep_boolean_mask]  # Remove the inliers in the mask accordingly
        #print("REMOVED: {}".format(correspondences.shape[0] - clean_correspondences.shape[0]))
        # Retrieve coordinates from correspondences
        s_c = source.points[clean_correspondences[:, 0]]  # [num_corresp, 3]
        t_c = target.points[clean_correspondences[:, 1]]  # [num_corresp, 3]
        beta = np.repeat(beta_i, s_c.shape[0])  # [num_corresp, 1]

        # Compute invariants
        if inlier_mask is not None:
            trim_s, trim_t, alpha, _, pair_inlier_mask = compute_trims(s_c, t_c, beta, kept_inlier_mask)
        else:
            trim_s, trim_t, alpha, _ = compute_trims(s_c, t_c, beta)
        tim_s, tim_t, delta, pairs = compute_tims(s_c, t_c, beta)
        #reg_res.start_pair_inlier_mask = pair_inlier_mask  # Storing inliers for statistics
        USE_ROT_CPU = False
        # Estimate scale
        if 's' in variables:
            print("\n------------Estimating scale------------")
            reg_res, est_inlier_inds = self._estimate_scale(trim_s=trim_s, trim_t=trim_t, alpha=alpha, c=c, reg_res=reg_res)

            # Transform source with estimate, if given
            s_c = reg_res.s * s_c
            tim_s = reg_res.s * tim_s

            # Remove scale outlier correspondences
            tim_s, tim_t, delta = tim_s[est_inlier_inds], tim_t[est_inlier_inds], delta[est_inlier_inds]
            #reg_res.est_scale_inlier_mask = pair_inlier_mask[est_inlier_inds]  # estimated inliers after scale
            pairs = pairs[est_inlier_inds]  # Correpondence pairs, that are considered inliers
            print(f"> Kept {est_inlier_inds.shape[0]} | {trim_s.shape[0]}")
        else:
            print("\n-------Running MC Scale Inlier -------")
            est_inlier_inds = mc_inlier_scale(s_c, t_c, beta, c)
            print("\n Kept {} of {} Correspondences [{}]".format(est_inlier_inds.shape[0], s_c.shape[0], est_inlier_inds.shape[0] / s_c.shape[0]))
            reg_res.s = 1
            if est_inlier_inds.shape[0] > 95:
                #USE_ROT_CPU = True
                print("Too big, removing inds")
                est_inlier_inds = est_inlier_inds[:95]  # ToDo: Find a better solution to this

            #kept_inlier_mask = kept_inlier_mask[est_inlier_inds]
            if est_inlier_inds.shape[0] < 4:
                print("> Trying again!")
                est_inlier_inds = mc_inlier_scale(s_c, t_c, beta = np.repeat(0.8, s_c.shape[0]), c=c)
                print("\n Kept {} of {} Correspondences [{}]".format(est_inlier_inds.shape[0], s_c.shape[0],
                                                                     est_inlier_inds.shape[0] / s_c.shape[0]))
                if est_inlier_inds.shape[0] < 4:
                    est_inlier_inds = mc_inlier_scale(s_c, t_c, beta=np.repeat(10, s_c.shape[0]), c=c)
                    print("\n Kept {} of {} Correspondences [{}]".format(est_inlier_inds.shape[0], s_c.shape[0],
                                                                         est_inlier_inds.shape[0] / s_c.shape[0]))
            s_c, t_c, beta = s_c[est_inlier_inds], t_c[est_inlier_inds], beta[est_inlier_inds]

            if inlier_mask is not None:
                tim_s, tim_t, delta, pairs, pair_inlier_mask = compute_tims(s_c, t_c, beta, kept_inlier_mask)
            else:
                tim_s, tim_t, delta, pairs = compute_tims(s_c, t_c, beta)

        if 'R' in variables:
            print("\n------------Estimating rotation------------")
            reg_res, est_inlier_inds = self._estimate_rotation(tim_s=tim_s, tim_t=tim_t, delta=delta, c=c, reg_res=reg_res, use_cpu=USE_ROT_CPU)

            # Transform source with estimate, if given
            s_c = (reg_res.R @ s_c.transpose()).transpose()
        #else:
        #    print("\n-------Running MC Scale Inlier -------")
        #    # ToDo: This is not done
        #    est_inlier_inds = mc_inlier_rotation(s_c, t_c, beta, c)
        #    raise NotImplementedError

        pairs = pairs[est_inlier_inds]  # Correspondence pairs, that are inliers w.r.t. scale and rotation
        single, single_ind = np.unique(pairs.flatten(), return_index=True)  # ToDo: Not sure about this, probably wrong
        #reg_res.est_rotation_inlier_mask = reg_res.est_scale_inlier_mask[est_inlier_inds]

        s_c, t_c = s_c[single], t_c[single]  # Extract inlier points in scale and rotation
        beta = beta[single]  # Extract inlier betas in scale and rotation
        #single_mask = np.repeat(reg_res.est_rotation_inlier_mask[:, None], 2, axis=1).flatten()[single_ind]

        if 't' in variables:
            print("\n------------Estimating translation------------")
            reg_res, est_inlier_inds = self._estimate_translation(source_c=s_c, target_c=t_c, beta=beta, c=c, reg_res=reg_res)

        else:
            est_inlier_inds = np.arange(s_c.shape[0])
            # ToDo: only remove outliers

        #reg_res.est_translation_inlier_mask = single_mask[est_inlier_inds]

        ## ----------- Temp -------------
        ## create trims and compute scale
        #if 's' in variables:
        #    trim_s, trim_t, alpha, _ = compute_trims(s_c, t_c, beta)
        #    s_hat = compute_scale_estimate(trim_s, trim_t, alpha)
        #    reg_res.s = s_hat * reg_res.s

        #print("Previous scale estimate: {}".format(reg_res.s))
        #print("New scale estimate: {}".format(s_hat*reg_res.s))

        return reg_res, est_inlier_inds

    def _estimate_scale(self, trim_s, trim_t, alpha, c, reg_res):

        t0 = time.time()

        # Create fitting_graph
        edges = con_func_scale_3d(trim_s, trim_t, alpha, c)
        fitting_graph = Graph(edges, trim_s.shape[0])

        # Solve FittingGraph
        inliers = fitting_graph.cliques()

        # If there are multiple largest cliques, find the one that has the lowest fitting error
        errs = []
        for clique in inliers:
            trim_inlier_s = trim_s[np.array(clique)]
            trim_inlier_t = trim_t[np.array(clique)]
            alpha_inlier = alpha[np.array(clique)]

            s_hat, res = compute_scale_estimate(trim_inlier_s, trim_inlier_t, alpha_inlier, return_residual=True)

            errs.append(res)

        inlier_inds = inliers[np.argmin(errs)]
        alpha_inlier = alpha[np.array(inlier_inds)]
        trim_inlier_s = trim_s[np.array(inlier_inds)]
        trim_inlier_t = trim_t[np.array(inlier_inds)]

        # return scale estimate
        s_hat, res = compute_scale_estimate(trim_inlier_s, trim_inlier_t, alpha_inlier, return_residual=True)
        reg_res.s = s_hat
        reg_res.d_time_s = time.time() - t0

        return reg_res, inlier_inds

    def _estimate_rotation(self, tim_s, tim_t, delta, c, reg_res, use_cpu):

        t0 = time.time()

        # Create fitting graph
        if use_cpu:
            print("Using CPU for rotation consensus.")
            edges = con_func_rotation_3d(tim_s, tim_t, delta, c)
        else:
            edges = cp_con_func_rotation_3d(tim_s, tim_t, delta, c)
        print("Searching for rotation inlier set")
        fitting_graph = Graph(edges, tim_s.shape[0])

        # Solve Fitting Graph
        inliers = fitting_graph.cliques()

        # Find the maximal clique that will lead to the lowest fitting error
        errs = []
        for clique in inliers:

            inlier_tim_s = tim_s[np.array(clique)]
            inlier_tim_t = tim_t[np.array(clique)]
            inlier_delta = delta[np.array(clique)]

            R_hat, res = compute_rotation_estimate(inlier_tim_s, inlier_tim_t, inlier_delta, return_residual=True)

            errs.append(res)

        # Compute rotation estimate
        inlier_clique = inliers[np.argmin(errs)]

        inlier_tim_s = tim_s[np.array(inlier_clique)]
        inlier_tim_t = tim_t[np.array(inlier_clique)]
        inlier_delta = delta[np.array(inlier_clique)]

        reg_res.R = compute_rotation_estimate(inlier_tim_s, inlier_tim_t, inlier_delta)
        reg_res.d_time_R = time.time() - t0

        return reg_res, inlier_clique

    def _estimate_translation(self, source_c, target_c, beta, c, reg_res):

        t0 = time.time()

        # Create fitting graph
        edges = con_func_translation_3d(source_c, target_c, beta, c)
        fitting_graph = Graph(edges, source_c.shape[0])

        # Solve Fitting Graph
        inliers = fitting_graph.cliques()

        errs = []
        for clique in inliers:

            source_inliers = source_c[np.array(clique)]
            target_inliers = target_c[np.array(clique)]
            beta_inliers = beta[np.array(clique)]

            t_hat, res = compute_translation_estimate(source_inliers, target_inliers, beta_inliers, return_residual=True)

            errs.append(res)

        inlier_clique = inliers[np.argmin(errs)]
        source_inliers = source_c[np.array(inlier_clique)]
        target_inliers = target_c[np.array(inlier_clique)]
        beta_inliers = beta[np.array(inlier_clique)]

        if len(inlier_clique) > 0:
            t_hat = compute_translation_estimate(source_inliers, target_inliers, beta_inliers)
        else:
            t_hat = (target_c - source_c).mean(axis=0)

        reg_res.t = t_hat
        reg_res.d_time_t = time.time() - t0

        return reg_res, inlier_clique


if __name__ == "__main__":

    parameters = {'FG_REG': {'threshold': 1, 'beta_i': 0.0554}}

    dataset = StanfordDataset('../datasets/data/stanford_dataset')
    dataset.exclude_transform = []
    samp = dataset.get_sample()

    inlier_c, outlier_c = samp.sample_correspondences(0.9, 30)
    correspondences = np.concatenate([inlier_c, outlier_c], axis=0)
    print(correspondences)
    np.ones(inlier_c.shape[0])
    inlier_mask = np.concatenate([np.ones(inlier_c.shape[0]), np.zeros([outlier_c.shape[0]])])

    fg_reg = FittingGraphRegistration(parameters)
    res, _ = fg_reg(samp.source, samp.target, correspondences, variables=['R', 's', 't'], inlier_mask=inlier_mask)

    #print(res.get_inlier_percentages())
    #print(res.get_inlier_retention_percentages())#
    print(res.s)
    print(samp.s)
    print(res.R)
    print(samp.R)
    print(res.t)
    print(samp.t)

    
    #print(samp.str_gt_transform())
