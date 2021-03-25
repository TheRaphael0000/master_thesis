"""Rank list fusion module."""

import s_curves
import distances
from corpus import brunet, oxquarry, st_jean
from linking import compute_links
from collections import defaultdict
import numpy as np
from evaluate import evaluate_linking


def rank_list_fusion(rank_lists, s_curve, args={}):
    # grouping same links
    grouped_by_link = defaultdict(list)
    for rank_list in rank_lists:
        x, y = s_curve(len(rank_list), **args)
        for i, (link, dist) in enumerate(rank_list):
            grouped_by_link[link].append(y[i])
    # average
    for k in grouped_by_link:
        grouped_by_link[k] = np.mean(grouped_by_link[k])
    overall_ranklist = list(dict(grouped_by_link).items())
    overall_ranklist.sort(key=lambda x: x[-1])
    return overall_ranklist


def compute_multiple_links(experiments, s_curve):
    rank_lists = []
    for exp in experiments:
        rank_list = compute_links(*exp)
        rank_lists.append(rank_list)
    if s_curve is None:
        return None, rank_lists
    rank_list_overall = rank_list_fusion(rank_lists, s_curve)
    return rank_list_overall, rank_lists


if __name__ == '__main__':
    _, X2, X, Y = brunet.parse()

    experiments = [
        [X, 5, 500, True, 0.1, distances.manhattan],
        [X, 0, 500, True, 0.1, distances.manhattan],
        [X2, 5, 500, True, 0.1, distances.manhattan],
        [X2, 0, 500, True, 0.1, distances.manhattan],
    ]
    s_curve = s_curves.sigmoid_reciprocal

    rank_list_overall, rank_lists = compute_multiple_links(experiments, s_curve)
    print("AP RPrec HPrec (Used for overall)")
    for rank_list in rank_lists:
        mesures = evaluate_linking(rank_list, Y)
        print(*mesures)
    print("AP RPrec HPrec (Overall)")
    mesures = evaluate_linking(rank_list_overall, Y)
    print(*mesures)
