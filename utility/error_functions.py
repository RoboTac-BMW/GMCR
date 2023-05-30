import copy

import numpy as np


def scale_error(reg_res, reg_pair):
    """
    Computes the error in scale.
    :param reg_res: instance of RegistrationResult (with scale not None)
    :param reg_pair: instance of PointCloudPair (with scale not None)
    :return: |s_hat - s|
    """
    return np.abs(reg_res.s - reg_pair.s)


def rotation_error(reg_res, reg_pair):
    """
    Computes the rotation error as geodesic distance.
    :param reg_res: instance of RegistrationResult (with rotation not None)
    :param reg_pair: instance of PointCloudPair (with rotation not None)
    :return: |arccos( (trace( R_hat^T @ R ) - 1) / 2 )|
    """
    R_hat = reg_res.R
    R = reg_pair.R

    inner = (np.trace(R_hat.transpose()@R) - 1) / 2
    error = np.abs(np.arccos(np.clip(inner, -1.0, 1.0)))

    return error


def translation_error(reg_res, reg_pair):
    """
    Computes the translation error as l2-norm.
    :param reg_res: instance of RegistrationResult (with translation not None)
    :param reg_pair: instance of PointCloudPair (with translation not None)
    :return: || t_hat - t ||
    """
    error = np.linalg.norm(reg_res.t - reg_pair.t)
    return error


def combined_error(reg_res, reg_pair, inlier_corresp):
    """
    Computes the scale, rotation and translation error as defined above.
    :param reg_res: instance of RegistrationResult (with scale, rotation, translation not None)
    :param reg_pair: instance of PointCloudPair (with scale, rotation, translation not None)
    :return: dict with {'e_s': error_scale, 'e_R': error_rotation, 'e_t': error_translation)
    """
    if reg_res.s is None:
        e_s = None
        e_adi = None
    else:
        e_s = scale_error(reg_res, reg_pair)
        e_adi = adi_known_correspondences(reg_res, reg_pair, inlier_corresp)
    e_R = rotation_error(reg_res, reg_pair)
    e_t = translation_error(reg_res, reg_pair)

    return {'e_s': e_s, 'e_R': e_R, 'e_t': e_t, 'e_adi': e_adi}


def adi_known_correspondences(reg_res, reg_pair, inlier_corresp):

    _reg_pair = copy.deepcopy(reg_pair)
    _reg_res = copy.deepcopy(reg_res)

    _reg_pair.source.points = _reg_res.s * _reg_pair.source.points
    _reg_pair.transform_source(_reg_res.R, _reg_res.t)

    c_source, c_target = _reg_pair.source.points[inlier_corresp[:, 0]], _reg_pair.target.points[inlier_corresp[:, 1]]

    adi = np.linalg.norm(c_target - c_source, axis=1).mean()

    return adi


def adi_unkown_correspondences(reg_res, reg_pair, src_pcd):

    src_est = copy.deepcopy(src_pcd)
    src_est.points = src_est.points * reg_res.s
    src_est.transform(reg_res.R, reg_res.t)

    src_gt = copy.deepcopy(src_pcd)
    src_gt.points = src_gt.points * reg_pair.s
    src_gt.transform(reg_pair.R, reg_pair.t)

    adi = np.linalg.norm(src_gt.points - src_est.points, axis=1).mean()

    return adi