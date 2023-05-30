import numpy as np
from fitting_graphs.src.invariants import compute_tims, compute_trims
from fitting_graphs.utility.graph import Graph


def remove_conflicting_correspondences(correspondences: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Removes correspondences, where the start- or end-point occurs twice
    Args:
        correspondences (np.ndarray): indices where [i, 0] is source and [i, 1] is target[N, 2]

    Returns:
        clean_correspondences (np.ndarray):  correspondences, where k double occurrences are removed [(N-k),2]
    """

    u_source, inv_s, count_s = np.unique(correspondences[:, 0], return_inverse=True, return_counts=True)
    source_counts = count_s[inv_s]
    keep_source = (source_counts == 1)

    u_target, inv_t, count_t = np.unique(correspondences[:, 1], return_inverse=True, return_counts=True)
    target_counts = count_t[inv_t]
    keep_target = (target_counts == 1)

    keep_overall = keep_source * keep_target
    print(f"> Kept {keep_overall.sum()} | {correspondences.shape[0]} correspondences")
    clean_correspondences = correspondences[(keep_overall == 1), :]

    return clean_correspondences, keep_overall


def mc_inlier_scale(source_points: np.ndarray, target_points: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:
    """
    Removes gross outliers in correspondences, based on the assumption that scale should be 1. Note that points
    in source and target should correspond to each other.
    Args:
        source_points (np.ndarray): points from source [N, 3]
        target_points (np.ndarray): points from target [N, 3]
        beta (np.ndarray): Noise bound [N, 1]
        c (float): Cut-off

    Returns:
        inliers (np.ndarray): indices for inliers in source_points and target_points [N, 1]
    """
    # ToDo: Verify this!
    trim_s, trim_t, alpha, pair_inds = compute_trims(source_points, target_points, beta)
    s = trim_t / trim_s
    edge_boolean = np.abs(s - 1) < (alpha * c)
    edges = pair_inds[edge_boolean]
    self_edges = np.array([np.arange(source_points.shape[0]), np.arange(source_points.shape[0])]).T
    edges = np.concatenate([edges, self_edges], axis=0)
    consistency_graph = Graph(edges, source_points.shape[0])
    inlier_clique = consistency_graph.cliques()[0]

    return inlier_clique


def mc_inlier_rotation(source_points: np.ndarray, target_points: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:
    """
    Removes gross outliers in correspondences by the assumption that rotation should be identity. Note that points
    in source and target should correspond to each other.
    Args:
        source_points (np.ndarray): points from source [N, 3]
        target_points (np.ndarray): points from target [N, 3]
        beta (np.ndarray): Noise bounds [N, 1]
        c (float): Cut-off

    Returns:
        inliers (np.ndarray): indices of inliers in source and target
    """
    tim_s, tim_t, delta, pair_inds = compute_tims(source_points, target_points, beta)

    edge_boolean = np.linalg.norm(tim_t - tim_s, axis=1) < c * delta
    edges = pair_inds[edge_boolean]
    self_edges = np.array([np.arange(source_points.shape[0]), np.arange(source_points.shape[0])]).T
    edges = np.concatenate([edges, self_edges], axis=0)
    cons_graph = Graph(edges, source_points.shape[0])

    inliers = cons_graph.cliques()[0]
    return inliers


# ToDo: how can we do this?
"""
def mc_inlier_translation(source_points: np.ndarray, target_points: np.ndarray, beta: np.ndarray,
                          c: float) -> np.ndarray:
    edge_boolean = np.linalg.norm(target_points - source_points, axis=1) < c * beta
    edges = pair_inds[edge_boolean]
    self_edges = np.array([np.arange(source_points.shape[0]), np.arange(source_points.shape[0])]).T
    edges = np.concatenate([edges, self_edges], axis=0)
    cons_graph = Graph(edges, source_points.shape[0])

    inliers = cons_graph.cliques()[0]
    return inliers
"""