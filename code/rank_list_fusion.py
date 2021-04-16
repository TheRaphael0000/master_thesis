"""Rank list fusion module."""

import s_curves
import distances
from corpus import brunet, oxquarry, st_jean
from linking import compute_links
from collections import defaultdict
import numpy as np
import scipy as sp
from evaluate import evaluate_linking


def rank_list_fusion(links, scores):
    n_lists = links.shape[0]
    n_links = links.shape[1]

    # grouping same links
    grp_by_link = defaultdict(list)
    for list_indice in range(n_lists):
        for link_indice in range(n_links):
            link = links[list_indice, link_indice]
            score = scores[list_indice, link_indice]
            grp_by_link[tuple(link)].append(score)
    # averaging
    for k in grp_by_link:
        grp_by_link[k] = np.mean(grp_by_link[k])
    rl = list(dict(grp_by_link).items())
    rl.sort(key=lambda x: x[-1])
    return rl


def fusion_raw_score(rank_lists):
    links = np.array([[link for link, score in rl] for rl in rank_lists])
    scores = np.array([[score for link, score in rl] for rl in rank_lists])
    return rank_list_fusion(links, scores)


def fusion_z_score(rank_lists):
    links = np.array([[link for link, score in rl] for rl in rank_lists])
    scores = np.array([[score for link, score in rl] for rl in rank_lists])
    zscores = sp.stats.zscore(scores, axis=1)
    return rank_list_fusion(links, zscores)


def fusion_s_curve_score(rank_lists, s_curve):
    links = np.array([[link for link, score in rl] for rl in rank_lists])
    x, y = s_curve(len(rank_lists[0]))
    s_curve_scores = np.array([np.array(y) for rl in rank_lists])
    return rank_list_fusion(links, s_curve_scores)


if __name__ == '__main__':
    _, X2, X, Y = brunet.parse()

    experiments = [
        [X, 5, 500, True, 0.1, distances.manhattan],
        [X, 0, 500, True, 0.1, distances.manhattan],
        [X2, 5, 500, True, 0.1, distances.cosine_distance],
        [X2, 0, 500, True, 0.1, distances.cosine_distance],
    ]
    s_curve = s_curves.sigmoid_reciprocal()

    rls = [compute_links(*e) for e in experiments]
    print("Rank lists")
    for rl in rls:
        print(*evaluate_linking(rl, Y))

    print("Raw score")
    rl_overall = fusion_raw_score(rls)
    print(*evaluate_linking(rl_overall, Y))
    print("Z-score")
    rl_overall = fusion_z_score(rls)
    print(*evaluate_linking(rl_overall, Y))
    print("S-curve")
    rl_overall = fusion_s_curve_score(rls, s_curve)
    print(*evaluate_linking(rl_overall, Y))
    exit()
