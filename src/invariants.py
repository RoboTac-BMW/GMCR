import numpy as np
from fitting_graphs.utility.functions import has_zero_vecs
import warnings


def compute_tims(source_points: np.ndarray, target_points: np.ndarray, beta: np.ndarray,
                 inlier_mask=None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Computes translation-invariant measurements by using pairs of measurements in source and target point cloud
    Args:
        source_points (): points with shape [N, 3]
        target_points (): points with shape [N, 3]
        beta (): Noise bounds with shape [N, 1]

    Returns:
        (shape changes if remove_double == True)
        tim_source: vectors between source points [N*N, 3]
        tim_target: vectors between target points [N*N, 3]
        delta: noise bounds [N*N, 1]
    """

    assert source_points.shape == target_points.shape

    x, y = np.triu_indices(source_points.shape[0], 1)
    pairs = np.array([x, y]).T.reshape(-1, 2)
    tim_source = source_points[x] - source_points[y]  # Get each pair of points in the source correspondences
    tim_target = target_points[x] - target_points[y]  # Get each pair of points in the target correspondences

    if has_zero_vecs(tim_target):
        warnings.warn("Encountered zero vector in target-tims")
    if has_zero_vecs(tim_source):
        warnings.warn("Encountered zero vector in source-tims")

    delta = beta[x] + beta[y]

    N = source_points.shape[0]
    assert tim_source.shape[0] == N*(N-1)/2

    if inlier_mask is not None:
        pair_inlier_mask = np.array([inlier_mask[x], inlier_mask[y]]).transpose().all(axis=1)
        return tim_source, tim_target, delta, pairs, pair_inlier_mask

    return tim_source, tim_target, delta, pairs


def compute_trims(source_points: np.ndarray, target_points: np.ndarray,
                  beta: np.ndarray, inlier_mask=None) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Computes the translation and rotation invariant measurements, by first computing tims on target and source and then
    applying the element-wise norm.
    Args:
        source_points (): points with shape [N, 3]
        target_points (): poinst with shape [N, 3]
        beta (): noise bounds [N, 1]

    Returns:
        (shape changes if remove_double==True)
        trim_source: trims for source points [N*N, 1]
        trim_target: trims for target points [N*N, 1]
        alpha: noise bound [N*N, 1]

    """
    if inlier_mask is not None:
        source_tims, target_tims, delta, pairs, pair_inlier_mask = compute_tims(source_points, target_points,
                                                                         beta, inlier_mask)
    else:
        source_tims, target_tims, delta, pairs = compute_tims(source_points, target_points, beta)

    trim_source = np.linalg.norm(source_tims, axis=1)
    trim_target = np.linalg.norm(target_tims, axis=1)
    alpha = delta / trim_source

    if has_zero_vecs(trim_source[:, None]):
        warnings.warn("Encountered zero value in source-trims")
    if has_zero_vecs(trim_target[:, None]):
        warnings.warn("Encountered zero value in target-trims")

    if inlier_mask is not None:
        return trim_source, trim_target, alpha, pairs, pair_inlier_mask
    else:
        return trim_source, trim_target, alpha, pairs


def compute_scale_estimate(trim_source: np.ndarray, trim_target: np.ndarray, alpha: np.ndarray, return_residual: bool = False) -> (float, float):
    """
    Computes the scale estimate for a set of inlier scale values.
    Args:
        trim_source (np.ndarray): translation and rotation invariant measurements for source [N_s, 1]
        trim_target (np.ndarray): translation and rotation invariant measurements for target [N_s, 1]
        alpha (np.ndarray): Noise bounds [N_s, 1]
        return_residual (bool): If true, returns the residual of the estimate as well
    Returns:
        s_hat (float): Scale estimate
        residual (float): Residual between measurements and estimate
    """

    s = trim_target / trim_source
    s_hat = (1 / ((1 / np.square(alpha)).sum())) * (s / np.square(alpha)).sum()

    if return_residual:
        residual = (np.square(s - s_hat) / np.square(alpha)).sum()
        return s_hat, residual
    else:
        return s_hat


def compute_rotation_estimate(tim_source: np.ndarray, tim_target: np.ndarray, delta: np.ndarray,
                              return_residual: bool = False) -> (np.ndarray, float):
    """
    Computes the rotation estimate given a set of inlier measurements.
    Args:
        tim_source (np.ndarray): translation invariant measurements from source [N_R, 3]
        tim_target (np.ndarray): translation invariant measurements from target [N_R, 3]
        delta (np.ndarray): Noise bound [N_R, 1]
        return_residual (bool): if true, will also return the residual for the estimate

    Returns:
        R_hat (np.ndarray): Rotation esimate as matrix [3, 3]
        residual (float): Residual between estimate and measurements
    """

    H = []
    for i in range(tim_source.shape[0]):
        H.append(tim_source[i, :][:, None] @ tim_target[i, :][None, :] / delta[i])

    H = np.array(H)

    H = H.sum(axis=0)

    u, s, vh = np.linalg.svd(H)

    R = vh.transpose() @ u.transpose()

    if return_residual:
        res = np.linalg.norm(tim_target - (R @ tim_source.transpose()).transpose(), axis=1).sum()  # ToDo: Seems like noise is missing
        return R, res
    else:
        return R


def compute_translation_estimate(source: np.ndarray, target: np.ndarray, beta: np.ndarray,
                                 return_residual: bool = False) -> (np.ndarray, float):
    """
    Computes the translation estimate given a set of inlier points.
    Args:
        source (np.ndarray): points from source [N_t, 3]
        target (np.ndarray): points from target [N_t, 3]
        beta (np.ndarray): Noise bound [N_t, 1]
        return_residual (bool): if true, also returns the residual for the estimate

    Returns:
        t_hat (np.ndarray): translation estimate [3, 1]
        res (float): residual for t_hat
    """
    t = target - source
    t_hat = (1 / (1 / beta[:, None]).sum()) * (t / beta[:, None]).sum(axis=0, keepdims=True)

    if return_residual:
        res = (np.square(t - t_hat) / np.square(beta[:, None])).sum()  # ToDo: Check if this is correct
        return t_hat, res
    else:
        return t_hat


