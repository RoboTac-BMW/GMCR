import open3d as o3d
import numpy as np
from fitting_graphs.base_classes import RegistrationMethod
from fitting_graphs.utility.registration import RegistrationResult


class FGRWrapper(RegistrationMethod):
    name = "fgr"

    def __init__(self, parameters):
        super(FGRWrapper, self).__init__(parameters)

    def __call__(self, source_pcd, target_pcd, correspondences, variables=None, inlier_inds=None):

        options = o3d.pipelines.registration.FastGlobalRegistrationOption()
        options.tuple_test = True
        options.use_absolute_scale = True
        options.maximum_correspondence_distance = self.parameters['max_corresp_dist']
        options.iteration_number = self.parameters['iterations']
        options.decrease_mu = True
        #options.division_factor = 1


        source = source_pcd.to_o3d()
        target = target_pcd.to_o3d()
        corresp = o3d.utility.Vector2iVector(correspondences)

        res = o3d.pipelines.registration.registration_fgr_based_on_correspondence(source, target, corresp, options)

        T = res.transformation
        R_s = T[0:3, 0:3]
        s = np.sqrt((R_s @ R_s.transpose())[0, 0])
        R = T[0:3, 0:3] / s
        t = T[:3, 3][None, :]

        reg_res = RegistrationResult(s, R, t)

        return reg_res, None
