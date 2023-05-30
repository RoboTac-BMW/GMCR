import numpy as np
import cupy as cp
from fitting_graphs.utility.math import cp_angle_between_vecs, cp_rot_two_vecs_angle, cp_rot_align_vecs, cp_cosine_law, cp_rodrigues_formula, cp_axis_between_vecs
from fitting_graphs.utility.math import angle_between_vecs, rot_two_vecs_angle, rot_align_vecs, cosine_law, axis_between_vecs, rodrigues_formula


def con_func_scale_3d(trim_source: np.ndarray, trim_target: np.ndarray, alpha: np.ndarray, c: float) -> np.ndarray:

    step_size = 10000000

    scale = cp.array(trim_target / trim_source)
    alpha = cp.array(alpha)
    c = cp.array(c)

    inds = np.array(np.triu_indices(scale.shape[0], k=0)).T.reshape(-1, 2)
    edges = None

    for i in range(0, inds.shape[0], step_size):

        ind_slice = cp.array(inds[i:(i + step_size), :])
        edge_boolean = (scale[ind_slice[:, 0]] - alpha[ind_slice[:, 0]] * c <= scale[ind_slice[:, 1]] + alpha[
            ind_slice[:, 1]] * c) * (scale[ind_slice[:, 1]] - alpha[ind_slice[:, 1]] * c <= scale[ind_slice[:, 0]] +
                                     alpha[ind_slice[:, 0]] * c)

        edge_slice = ind_slice[edge_boolean]

        if edges is None:
            edges = edge_slice
        else:
            edges = cp.concatenate([edges, edge_slice], axis=0)


    #other_side = copy.copy(edges)
    #other_side = np.flip(other_side, axis=1)

    #edges = np.concatenate([edges, other_side], axis=0)
    #selfcon = cp.array([cp.arange(trim_source.shape[0]), cp.arange(trim_source.shape[0])]).T
    #edges = cp.concatenate([edges, selfcon, selfcon], axis=0)
    return cp.asnumpy(edges)


def cp_con_func_rotation_3d(tim_source: np.ndarray, tim_target: np.ndarray, delta: np.ndarray, c: float) -> np.ndarray:

    # 1. Compute beta_k around each measurement (converting to angular representation)
    a = cp.array(tim_source)
    b = cp.array(tim_target)
    inds = cp.array(np.triu_indices(a.shape[0], k=1)).T.reshape(-1, 2)

    eps_k = cp.array(delta / c)
    norm_a = cp.linalg.norm(a, axis=1, keepdims=False)
    norm_b = cp.linalg.norm(b, axis=1, keepdims=False)
    beta_k = cp_cosine_law(a=eps_k, b=norm_b, c=norm_a)

    # 2. Scale a_i to the length of b_i
    a_scaled = a * (norm_b[:, None] / norm_a[:, None])

    # 3. Compute rotations for each a_k and b_k
    axes = cp_axis_between_vecs(a_scaled, b)
    angles = cp_angle_between_vecs(a_scaled, b)
    a_rot = cp_rodrigues_formula(axes[inds[:, 0]], angles[inds[:, 0]], a_scaled[inds[:, 1]])

    # 4. Compute angles in extreme points
    gamma = cp_angle_between_vecs(b[inds[:, 0]], a_rot)
    gamma_n = gamma - 2 * beta_k[inds[:, 0]]  # Angle between each b_k and B_k*a_i
    gamma_f = cp.pi - gamma - 2 * beta_k[inds[:, 0]]

    # 5. Create vectors a_f and a_n for each pair
    # Order was wrong previously!
    a_n = cp_rot_two_vecs_angle(b[inds[:, 0]], a_rot, beta_k[inds[:, 0]], b[inds[:, 0]])
    a_f = cp_rot_two_vecs_angle(a_rot, b[inds[:, 0]], beta_k[inds[:, 0]], b[inds[:, 0]])

    # 6. Construct Edges with consensus function
    edges_boolean = (gamma_f - cp_angle_between_vecs(b[inds[:, 1]], -a_f) < beta_k[inds[:, 1]])
    edges_boolean = edges_boolean * (gamma_n - cp_angle_between_vecs(b[inds[:, 1]], a_n) < beta_k[inds[:, 1]])
    edges = inds[edges_boolean]
    self_edges = cp.array([cp.arange(a.shape[0]), cp.arange(a.shape[0])]).T
    edges = cp.concatenate([edges, self_edges], axis=0)

    edges = cp.asnumpy(edges)
    return edges


def con_func_rotation_3d(tim_source: np.ndarray, tim_target: np.ndarray, delta: np.ndarray, c: float) -> np.ndarray:

    # 1. Compute beta_k around each measurement (converting to angular representation)
    a = np.array(tim_source)
    b = np.array(tim_target)
    inds = np.array(np.triu_indices(a.shape[0], k=1)).T.reshape(-1, 2)

    eps_k = np.array(delta / c)
    norm_a = np.linalg.norm(a, axis=1, keepdims=False)
    norm_b = np.linalg.norm(b, axis=1, keepdims=False)
    beta_k = cosine_law(a=eps_k, b=norm_b, c=norm_a)

    # 2. Scale a_i to the length of b_i
    a_scaled = a * (norm_b[:, None] / norm_a[:, None])

    # 3. Compute rotations for each a_k and b_k
    axes = axis_between_vecs(a_scaled, b)
    angles = angle_between_vecs(a_scaled, b)
    a_rot = rodrigues_formula(axes[inds[:, 0]], angles[inds[:, 0]], a_scaled[inds[:, 1]])

    # 4. Compute angles in extreme points
    gamma = angle_between_vecs(b[inds[:, 0]], a_rot)
    gamma_n = gamma - 2 * beta_k[inds[:, 0]]  # Angle between each b_k and B_k*a_i
    gamma_f = np.pi - gamma - 2 * beta_k[inds[:, 0]]

    # 5. Create vectors a_f and a_n for each pair
    # Order was wrong previously!
    a_n = rot_two_vecs_angle(b[inds[:, 0]], a_rot, beta_k[inds[:, 0]], b[inds[:, 0]])
    a_f = rot_two_vecs_angle(a_rot, b[inds[:, 0]], beta_k[inds[:, 0]], b[inds[:, 0]])

    # 6. Construct Edges with consensus function
    edges_boolean = (gamma_f - angle_between_vecs(b[inds[:, 1]], -a_f) < beta_k[inds[:, 1]])
    edges_boolean = edges_boolean * (gamma_n - angle_between_vecs(b[inds[:, 1]], a_n) < beta_k[inds[:, 1]])
    edges = inds[edges_boolean]
    self_edges = np.array([np.arange(a.shape[0]), np.arange(a.shape[0])]).T
    edges = np.concatenate([edges, self_edges], axis=0)

    return edges


def con_func_translation_3d(source: np.ndarray, target: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:

    t = target - source
    indices = np.array(np.triu_indices(t.shape[0], k=0)).T.reshape(-1, 2)
    norms = np.linalg.norm(t[indices[:, 0], :] - t[indices[:, 1], :], axis=1)
    beta_ij = beta[indices[:, 0]] + beta[indices[:, 1]]
    edges_boolean = (norms <= beta_ij / c)

    edges = indices[edges_boolean]

    return edges

