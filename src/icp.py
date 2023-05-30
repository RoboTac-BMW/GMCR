from fitting_graphs.base_classes import RegistrationMethod
import open3d as o3d
from fitting_graphs.utility.registration import RegistrationResult
import numpy as np


class ICPWrapper(RegistrationMethod):

    name = "icp"

    def __init__(self, parameters):
        super(ICPWrapper, self).__init__(parameters)

    def __call__(self, source_pcd, target_pcd, correspondences, variables=None, inlier_inds=None):

        source = source_pcd.to_o3d()
        target = target_pcd.to_o3d()
        threshold = self.parameters['thresh']
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
        criteria.max_iteration = self.parameters['max_it']
        criteria.relative_rmse = self.parameters['rel_rmse']
        criteria.relative_fitness = self.parameters['rel_fit']
        trans_init = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        res = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(res)
        T = res.transformation
        R_s = T[0:3, 0:3]
        s = np.sqrt((R_s @ R_s.transpose())[0, 0])
        R = T[0:3, 0:3] / s
        t = T[:3, 3][None, :]

        reg_res = RegistrationResult(s, R, t)

        return reg_res, None
