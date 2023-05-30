from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fitting_graphs.utility.functions import combine_over_objects, combine_over_ors


def visualize_outlier_benchmark(benchmark_path, date, ind, type):


    benchmark_path = Path(benchmark_path)
    scale_path = benchmark_path.joinpath(type)

    methods = [p.name for p in scale_path.iterdir()]
    results = {}

    for method in methods:

        path = benchmark_path.joinpath(type).joinpath(method).joinpath(
            '{}_{}_{}_{}.pkl'.format(str(date), str(ind), method, type))
        if path.exists():
            with path.open('rb') as f:
                results[method] = pickle.load(f)[1]

    outlier_ratios_left = list(list(results['RANSAC'].values())[0].keys())
    positions_left = [i - 0.2 for i in range(len(outlier_ratios_left))]
    combined_res_left = np.array(list(combine_over_objects(results['RANSAC']).values())).transpose()

    outlier_ratios_right = list(list(results['FG_REG'].values())[0].keys())
    positions_right = [i + 0.2 for i in range(len(outlier_ratios_right))]
    combined_res_right = np.array(list(combine_over_objects(results['FG_REG']).values())).transpose()


    fig, axs = plt.subplots(1, sharex=True, figsize=(15, 15))
    bp_1 = axs.boxplot(combined_res_left, sym='', positions=positions_left, widths=0.15, patch_artist=True,
                boxprops=dict(facecolor='red', color='red', alpha=0.5))
    bp_2 = axs.boxplot(combined_res_right, sym='', positions=positions_right, widths=0.15, patch_artist=True,
                boxprops=dict(facecolor='blue', color='blue', alpha=0.5))

    axs.set_yscale('log')
    axs.legend([bp_1["boxes"][0], bp_2["boxes"][0]], ['RANSAC', 'FG_REG'], loc='upper right')
    plt.xticks(range(0, len(outlier_ratios_left) + 0), outlier_ratios_left)
    plt.show()


def visualize_outlier_benchmark_complete(benchmark_path, date, ind, type):


    benchmark_path = Path(benchmark_path)
    scale_path = benchmark_path.joinpath(type)

    methods = [p.name for p in scale_path.iterdir()]
    results = {}

    for method in methods:

        path = benchmark_path.joinpath(type).joinpath(method).joinpath(
            '{}_{}_{}_{}.pkl'.format(str(date), str(ind), method, type))
        if path.exists():
            with path.open('rb') as f:
                results[method] = pickle.load(f)[1]

    outlier_ratios_left = list(list(results['RANSAC'].values())[0].keys())
    positions_left = [i - 0.2 for i in range(len(outlier_ratios_left))]
    combined_res_left = np.array(list(combine_over_objects(results['RANSAC']).values())).transpose()

    outlier_ratios_right = list(list(results['FG_REG'].values())[0].keys())
    positions_right = [i + 0.2 for i in range(len(outlier_ratios_right))]
    combined_res_right = np.array(list(combine_over_objects(results['FG_REG']).values())).transpose()

    outlier_ratios_middle = list(list(results['teaserpp'].values())[0].keys())
    positions_middle = [i for i in range(len(outlier_ratios_middle))]
    combined_res_middle = np.array(list(combine_over_objects(results['teaserpp']).values())).transpose()

    fig, axs = plt.subplots(1, sharex=True, figsize=(15, 15))
    bp_1 = axs.boxplot(combined_res_left, sym='', positions=positions_left, widths=0.15, patch_artist=True,
                boxprops=dict(facecolor='red', color='red', alpha=0.5))
    bp_2 = axs.boxplot(combined_res_right, sym='', positions=positions_right, widths=0.15, patch_artist=True,
                boxprops=dict(facecolor='blue', color='blue', alpha=0.5))

    bp_3 = axs.boxplot(combined_res_middle, sym='', positions=positions_middle, widths=0.15, patch_artist=True,
                       boxprops=dict(facecolor='green', color='green', alpha=0.5))

    axs.set_yscale('log')
    axs.legend([bp_1["boxes"][0], bp_2["boxes"][0], bp_3['boxes'][0]], ['RANSAC', 'FG_REG', 'TEASERPP'], loc='upper right')
    plt.xticks(range(0, len(outlier_ratios_left) + 0), outlier_ratios_left)
    plt.show()


def show_nan(benchmark_path, date, ind, type):

    benchmark_path = Path(benchmark_path)
    scale_path = benchmark_path.joinpath(type)

    methods = [p.name for p in scale_path.iterdir()]
    results = {}

    for method in methods:

        path = benchmark_path.joinpath(type).joinpath(method).joinpath(
            '{}_{}_{}_{}.pkl'.format(str(date), str(ind), method, type))
        if path.exists():
            with path.open('rb') as f:
                results[method] = pickle.load(f)[1]

    outlier_ratios = list(list(results['FG_REG'].values())[0].keys())

    combined_res = np.array(list(combine_over_objects(results['FG_REG']).values())).transpose()

    nan_list = []

    for i in range(len(outlier_ratios)):
        print(combined_res[:, i].shape)
        nan_list.append(np.count_nonzero(np.isnan(combined_res[:, i])) / combined_res[:, i].shape[0])

    plt.plot(range(len(nan_list)), nan_list)
    plt.show()


def show_nans_over_objects(benchmark_path, date, ind, type):

    benchmark_path = Path(benchmark_path)
    scale_path = benchmark_path.joinpath(type)

    methods = [p.name for p in scale_path.iterdir()]
    results = {}

    for method in methods:

        path = benchmark_path.joinpath(type).joinpath(method).joinpath(
            '{}_{}_{}_{}.pkl'.format(str(date), str(ind), method, type))
        if path.exists():
            with path.open('rb') as f:
                results[method] = pickle.load(f)[1]

    outlier_ratios = list(list(results['FG_REG'].values())[0].keys())

    for obj, obj_res in results['FG_REG'].items():

        or_nans = []
        for o_r, o_r_res in obj_res.items():
            or_nans.append(np.isnan(o_r_res).sum() / len(o_r_res))

        plt.plot(range(len(outlier_ratios)), or_nans, label=obj)

    plt.legend()
    plt.show()

def compare_runs_objects(run_1=None, run_2=None, run_3=None):

    fig, axs = plt.subplots(1, sharex=True, figsize=(15, 15))
    leg_list = []
    leg_names = []
    assert run_1 is not None

    if run_1 is not None:
        method_name_1, res_1 = run_1[0], run_1[1][1]
        obj_names_left = list(res_1.keys())
        positions_left = [i - 0.2 for i in range(len(obj_names_left))]
        combined_res_left = np.array(list(combine_over_ors(res_1).values())).transpose()
        bp_1 = axs.boxplot(combined_res_left, sym='', positions=positions_left, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='red', color='red', alpha=0.5))
        leg_list.append(bp_1["boxes"][0])
        leg_names.append(method_name_1)

    if run_2 is not None:
        method_name_2, res_2 = run_2[0], run_2[1][1]
        outlier_ratios_right = list(res_2.keys())
        positions_right = [i + 0.2 for i in range(len(outlier_ratios_right))]
        combined_res_right = np.array(list(combine_over_ors(res_2).values())).transpose()
        bp_2 = axs.boxplot(combined_res_right, sym='', positions=positions_right, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='blue', color='blue', alpha=0.5))
        leg_list.append(bp_2["boxes"][0])
        leg_names.append(method_name_2)

    if run_3 is not None:
        method_name_3, res_3 = run_3[0], run_3[1][1]
        outlier_ratios_middle = list(res_3.keys())
        positions_middle = [i for i in range(len(outlier_ratios_middle))]
        combined_res_middle = np.array(list(combine_over_ors(res_3).values())).transpose()
        bp_3 = axs.boxplot(combined_res_middle, sym='', positions=positions_middle, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='green', color='green', alpha=0.5))
        leg_list.append(bp_3["boxes"][0])
        leg_names.append(method_name_3)


    axs.set_yscale('log')
    axs.legend(leg_list, leg_names, loc='upper right')
    plt.xticks(range(0, len(obj_names_left) + 0), obj_names_left)
    plt.show()


def compare_runs(run_1=None, run_2=None, run_3=None):

    fig, axs = plt.subplots(1, sharex=True, figsize=(15, 15))
    leg_list = []
    leg_names = []
    assert run_1 is not None

    if run_1 is not None:
        method_name_1, res_1 = run_1[0], run_1[1][1]
        outlier_ratios_left = list(list(res_1.values())[0].keys())
        outlier_ratios_left.sort()
        positions_left = [i - 0.2 for i in range(len(outlier_ratios_left))]
        combined_res_left = np.array(list(combine_over_objects(res_1).values())).transpose()
        bp_1 = axs.boxplot(combined_res_left, sym='', positions=positions_left, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='red', color='red', alpha=0.5))
        leg_list.append(bp_1["boxes"][0])
        leg_names.append(method_name_1)

    if run_2 is not None:
        method_name_2, res_2 = run_2[0], run_2[1][1]
        outlier_ratios_right = list(list(res_2.values())[0].keys())
        positions_right = [i + 0.2 for i in range(len(outlier_ratios_right))]
        combined_res_right = np.array(list(combine_over_objects(res_2).values())).transpose()
        bp_2 = axs.boxplot(combined_res_right, sym='', positions=positions_right, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='blue', color='blue', alpha=0.5))
        leg_list.append(bp_2["boxes"][0])
        leg_names.append(method_name_2)

    if run_3 is not None:
        method_name_3, res_3 = run_3[0], run_3[1][1]
        outlier_ratios_middle = list(list(res_3.values())[0].keys())
        positions_middle = [i for i in range(len(outlier_ratios_middle))]
        combined_res_middle = np.array(list(combine_over_objects(res_3).values())).transpose()
        bp_3 = axs.boxplot(combined_res_middle, sym='', positions=positions_middle, widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor='green', color='green', alpha=0.5))
        leg_list.append(bp_3["boxes"][0])
        leg_names.append(method_name_3)


    axs.set_yscale('log')
    axs.legend(leg_list, leg_names, loc='upper right')
    plt.xticks(range(0, len(outlier_ratios_left) + 0), outlier_ratios_left)
    plt.show()


if __name__ == '__main__':

    PATH = '../results/5_known_scale_registration/'
    #PATH = "../results/2_complete_registration"
    #PATH = '../results/1_individual_registration'
    #visualize_outlier_benchmark(PATH, 20211231, 1, 'translation')
    #show_nan(PATH, 20210703, 1, 'translation')
    #show_nans_over_objects(PATH, 20210706, 1, 'rotation')
    #show_nan(PATH, 20210707, 1, 'rotation')


    visualize_outlier_benchmark_complete(PATH, 20220209, 1, 'translation')
    #visualize_outlier_benchmark_complete(PATH, 20220209, 1, 'translation')