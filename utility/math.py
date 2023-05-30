import numpy as np
from scipy.spatial.transform import Rotation as spR
import cupy as cp
from fitting_graphs.utility.timer import Timer
import time


def rodrigues_formula(rot_axes, angles, vec_rot):
    vecs = vec_rot * np.cos(angles[:, None]) + np.cross(rot_axes, vec_rot, axisa=1, axisb=1) * np.sin(angles[:, None])
    vecs += rot_axes * np.einsum('...j,...j', rot_axes, vec_rot)[:, None] * (1 - np.cos(angles[:, None]))
    return vecs


def cp_rodrigues_formula(rot_axes, angles, vec_rot):
    vecs = vec_rot * cp.cos(angles[:, None]) + cp.cross(rot_axes, vec_rot, axisa=1, axisb=1) * cp.sin(angles[:, None])
    vecs += rot_axes * cp.einsum('...j,...j', rot_axes, vec_rot)[:, None] * (1 - cp.cos(angles[:, None]))
    return vecs


def axis_between_vecs(vec1, vec2):
    timer = Timer()
    rot_axes = np.cross(vec1, vec2, axisa=1, axisb=1)
    #timer.log("inner 1")
    rot_axes = rot_axes / np.linalg.norm(rot_axes, axis=1, keepdims=True)
    #timer.log("inner 2")
    return rot_axes


def cp_axis_between_vecs(vec1, vec2):
    timer = Timer()
    rot_axes = cp.cross(vec1, vec2, axisa=1, axisb=1)
    #timer.log("inner 1")
    rot_axes = rot_axes / cp.linalg.norm(rot_axes, axis=1, keepdims=True)
    #timer.log("inner 2")
    return rot_axes


def angle_between_vecs(vec1, vec2):
    dot_prod = np.einsum('...j,...j', vec1, vec2)
    vec1_norm = np.linalg.norm(vec1, axis=1)
    vec2_norm = np.linalg.norm(vec2, axis=1)
    return np.arccos(np.clip(dot_prod / (vec1_norm * vec2_norm), -1.0, 1.0))


def cp_angle_between_vecs(vec1, vec2):
    dot_prod = cp.einsum('...j,...j', vec1, vec2)  # Is slow sometimes somehow
    #dot_prod = cp.diag(vec1 @ vec2.T)
    vec1_norm = cp.linalg.norm(vec1, axis=1)
    vec2_norm = cp.linalg.norm(vec2, axis=1)
    res = cp.arccos(cp.clip(dot_prod / (vec1_norm * vec2_norm), -1.0, 1.0))
    return res


def rot_two_vecs_angle(vec1, vec2, angle, vec_rot):
    rot_axes = np.cross(vec1, vec2, axisa=1, axisb=1)
    rot_axes = rot_axes / np.linalg.norm(rot_axes, axis=1, keepdims=True)

    vecs = vec_rot * np.cos(angle[:, None]) + np.cross(rot_axes, vec_rot, axisa=1, axisb=1) * np.sin(angle[:, None])
    vecs += rot_axes * np.einsum('...j,...j', rot_axes, vec_rot)[:, None] * (1 - np.cos(angle[:, None]))
    return vecs


def cp_rot_two_vecs_angle(vec1, vec2, angle, vec_rot):
    rot_axes = cp.cross(vec1, vec2, axisa=1, axisb=1)
    rot_axes = rot_axes / cp.linalg.norm(rot_axes, axis=1, keepdims=True)

    vecs = vec_rot * cp.cos(angle[:, None]) + cp.cross(rot_axes, vec_rot, axisa=1, axisb=1) * cp.sin(angle[:, None])
    vecs += rot_axes * cp.einsum('...j,...j', rot_axes, vec_rot)[:, None] * (1 - cp.cos(angle[:, None]))
    return vecs


def rot_align_vecs(vec1, vec2):
    # compute axis of rotation
    rot_axes = np.cross(vec1, vec2, axisa=1, axisb=1)
    rot_angle = angle_between_vecs(vec1, vec2)

    # Convert to axis angle
    rot_vecs = (rot_axes / np.linalg.norm(rot_axes, axis=1, keepdims=True)) * rot_angle[:, None]
    rotation = spR.from_rotvec(rot_vecs)
    return rotation.as_matrix()


def cp_rot_align_vecs(vec1, vec2, vec_rot):
    # compute axis of rotation
    rot_axes = cp.cross(vec1, vec2, axisa=1, axisb=1)
    rot_angle = angle_between_vecs(vec1, vec2)
    vecs = vec_rot * cp.cos(rot_angle[:, None]) + cp.cross(rot_axes, vec_rot, axisa=1, axisb=1) * cp.sin(rot_angle[:, None])
    vecs += rot_axes * cp.einsum('...j,...j', rot_axes, vec_rot)[:, None] * (1 - cp.cos(rot_angle[:, None]))
    return vecs
    

def cosine_law(a, b, c):
    inner = (np.square(b) + np.square(c) - np.square(a)) / (2 * b * c)
    return np.arccos(np.clip(inner, -1.0, 1.0))
    beta_k = cosine_law(a=eps_k, b=norm_b, c=norm_a)

def cp_cosine_law(a, b, c):
    inner = (cp.square(b) + cp.square(c) - cp.square(a)) / (2 * b * c)
    return cp.arccos(cp.clip(inner, -1.0, 1.0))
