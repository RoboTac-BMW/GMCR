import yaml
from pathlib import Path
from datetime import datetime
import copy
import numpy as np
from sys import getsizeof
import pickle
from scipy.spatial.transform import Rotation

def load_yaml_file(file_path):

    with open(file_path, 'r') as f:

        content = yaml.safe_load(f)

    return content


def load_pickle_file(file_path):

    with open(file_path, 'rb') as file:

        content = pickle.load(file)

    return content


def write_pickle_file(file_path, obj):

    with open(file_path, 'wb') as file:

        pickle.dump(obj, file)


def get_next_file_prefix(path):
    """
    Given a path it will get the next prefix, which consists of date and index of the run.
    :param path: pathlib.Path to the folder, which contains the files
    :return: str of form YYYYMMDD_{next_ind}
    """
    # Get all file names
    file_names = list(path.iterdir())

    # Get tuples of prefixes
    prefixes = [(str(p.name).split('_')[0], str(p.name).split('_')[1]) for p in file_names]

    # Get the tuples that correspond to today
    today = datetime.now().strftime('%Y%m%d')
    today_prefixes = [pref for pref in prefixes if int(pref[0]) == int(today)]

    new_date = str(today)  # Date should always be today

    if not today_prefixes:
        # no prefix from today, so use the date today and the index 1
        new_index = 1
    else:
        # Get highest index
        indices = [int(pref[1]) for pref in today_prefixes]
        indices.sort()
        new_index = str(indices[-1] + 1)

    return "{}_{}".format(new_date, new_index)


def combine_over_objects(result):
    """
    Combines an error dict over multiple objects, where errors corresponding to the same outlier ratio from
    all objects are stacked into a list.
    """
    combined_res = {}

    for obj_res in result.values():

        o_ors = list(obj_res.keys())
        o_ors.sort()

        for o_r in o_ors:

            if isinstance(obj_res[o_r], list):
                if o_r in combined_res.keys():
                    combined_res[o_r] += obj_res[o_r]
                else:
                    combined_res[o_r] = copy.deepcopy(obj_res[o_r])

            elif isinstance(obj_res[o_r], dict):
                if o_r in combined_res.keys():

                    for key, val in obj_res[o_r].items():
                        combined_res[o_r][key] += val

                else:
                    combined_res[o_r] = copy.deepcopy(obj_res[o_r])


            else:
                raise NotImplementedError

    return combined_res


def combine_over_ors(result):
    """
    """
    combined_res = {}

    for obj_name, obj_res in result.items():

        for o_r in obj_res.keys():

            if obj_name in combined_res.keys():
                combined_res[obj_name] += obj_res[o_r]
            else:
                combined_res[obj_name] = copy.deepcopy(obj_res[o_r])

    return combined_res

def has_zero_vecs(mat):

    assert len(mat.shape) == 2

    zero_vec = np.zeros((1, mat.shape[1]))
    mat_zeros = (mat == zero_vec).all(axis=1)
    sdf = mat_zeros.any()
    a = np.where(mat_zeros)
    return mat_zeros.any()


def str_size_of_array_MB(array):
    return round(getsizeof(array) / 1024 / 1024, 2)


def edge_set_equivalent(edges_a, edges_b):
    set_a = set(tuple(map(tuple, edges_a)))
    set_b = set(tuple(map(tuple, edges_b)))

    return set_a == set_b


def sym_edge_set_equivalent(edges_a, edges_b):
    set_a = set(tuple(map(tuple, np.sort(edges_a, axis=1))))
    set_b = set(tuple(map(tuple, np.sort(edges_b, axis=1))))

    diff = (set_a - set_b).union(set_b - set_a)
    if diff:
        print("Different edges are {}".format(diff))

    return set_a == set_b


def sym_find_duplicate_edges(edges):

    sorted_edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    return unique_edges[counts > 1]


def quaternion_to_rot(quat):

    r = Rotation.from_quat(quat)
    print(r.as_euler('zxy'))
    return r.as_matrix()


def invert_transform_sep(R, t):

    R_inv = np.linalg.inv(R)
    t_inv = (-1) * R_inv@t

    return R_inv, t_inv


def invert_transform(T):

    R = T[0:3, 0:3]
    t = T[0:3, 3]

    R_inv, t_inv = invert_transform_sep(R, t)

    T_inv = np.zeros_like(T)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    T_inv[3, 3] = 1

    return T_inv


def to_transform(R, t):

    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1

    return T


def from_transform(T):

    R = T[0:3, 0:3]
    t = T[0:3, 3]

    return R, t