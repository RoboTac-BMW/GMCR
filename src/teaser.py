import sys
sys.path.append('/home/mgentner/repos/MT_CertifiableRobustPCR')

import teaserpp_python as tpp
from fitting_graphs.base_classes import RegistrationMethod
from fitting_graphs.utility.registration import RegistrationResult
NOISE_BOUND = 0.05


class TEASERWrapper(RegistrationMethod):

    name = "teaserpp"

    def __init__(self, parameters):

        super(TEASERWrapper, self).__init__(parameters)

    def __call__(self, source_pcd, target_pcd, correspondences, variables=None, inlier_inds=None):


        if 's' in variables:
            # estimate scale
            pass

        if 'R' in variables:
            # estimate rotation
            pass

        if 't' in variables:
            # estimate translation
            pass

        solver_params = tpp.RobustRegistrationSolver.Params()
        solver_params.cbar2 = self.parameters['c_bar']
        solver_params.noise_bound = self.parameters['beta_i']
        solver_params.rotation_tim_graph = tpp.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.COMPLETE
        solver_params.inlier_selection_mode = tpp.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        if 's' in variables:
            solver_params.estimate_scaling = True
        else:
            solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = tpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 10000
        solver_params.rotation_cost_threshold = 1e-12

        solver = tpp.RobustRegistrationSolver(solver_params)
        solver.solve(source_pcd.points[correspondences[:, 0]].transpose(),
                     target_pcd.points[correspondences[:, 1]].transpose())

        solution = solver.getSolution()

        s = solution.scale
        R = solution.rotation
        t = solution.translation

        reg_res = RegistrationResult()
        reg_res.s = s
        reg_res.R = R
        reg_res.t = t[None, :]

        return reg_res, None

if __name__ == "__main__":

   teaser = TEASERWrapper({'teaserpp': {}})