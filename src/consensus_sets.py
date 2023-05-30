import numpy as np


def in_scale_consensus_set(scale_measurements, scale_estimate, alpha, c):
    """
    Given a scale estimate and a set of measurements, scale_consensus_set returns the set of measurements that
    is considered an inlier for this estimate.
    Args:
        scale_measurements:
        scale_estimate:

    Returns:

    """
    in_consensus_set = (np.square(scale_estimate - scale_measurements) / np.square(alpha) <= np.square(1 / c))

    return in_consensus_set


def in_rotation_consensus_set(rotation_measurements, rotation_estimate):

    pass


def in_translation_consensus_set(translation_measurements, translation_estimate):

    pass