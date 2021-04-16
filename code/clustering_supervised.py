"""Clustering module."""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import distances
import s_curves
from corpus import oxquarry
from corpus import brunet
from corpus import st_jean
from corpus import pan16
from linking import compute_links
from rank_list_fusion import fusion_s_curve_score
from misc import dataset_infos
from misc import distances_matrix_from_rank_list
from misc import rank_list_to_txt
from evaluate import evaluate_linking
from evaluate import evaluate_clustering


def clustering(rank_list, clf):
    print(" -- Clustering -- ")
    distance_threshold = compute_distance_threshold(rank_list, clf)
    print("distance_threshold:", distance_threshold)
    args = {
        "n_clusters": None,
        "affinity": "precomputed",
        "linkage": "average",
        "distance_threshold": distance_threshold
    }
    ac = AgglomerativeClustering(**args)
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    ac.fit(distances_matrix)
    return ac


def clustering_eval(Y_pred, Y):
    b3_prec, b3_rec, b3_fscore, mis = evaluate_clustering(Y_pred, Y)
    print("bcubed.precision", b3_prec)
    print("bcubed.recall", b3_rec)
    print("bcubed.fscore", b3_fscore)
    print("adjusted_mutual_info_score", mis)


def rank_list_feature_extraction(rank_list):
    X = [[np.log((i + 1) / len(rank_list)), score]
         for i, (link, score) in enumerate(rank_list)]
    return X


def compute_distance_threshold(rank_list, clf):
    X = rank_list_feature_extraction(rank_list)
    Y_pred = clf.predict(X)

    # the sum give the position n of the "flip" in the rank list since
    # the n first should be ones
    i = np.sum(Y_pred)
    distance_threshold = rank_list[i][-1]
    return distance_threshold


def experiments(Xi):
    return [
        [Xi, 0, 500, True, 1e-1, distances.manhattan],
        [Xi, 0, 500, False, 1e-1, distances.tanimoto],
        [Xi, 0, 500, False, 1e-1, distances.clark],
        [Xi, 0, 500, False, 1e-1, distances.matusita],
        [Xi, 0, 500, True, 1e-1, distances.cosine_distance],

        [Xi, 6, 500, True, 1e-1, distances.manhattan],
        # [Xi, 6, 500, False, 1e-1, distances.tanimoto],
        # [Xi, 6, 500, False, 1e-1, distances.clark],
        # [Xi, 6, 500, False, 1e-1, distances.matusita],
        [Xi, 6, 500, True, 1e-1, distances.cosine_distance],
    ]


def linking(X):
    print(" -- Linking -- ")
    experiments_ = experiments(X)
    s_curve = s_curves.sigmoid_reciprocal()
    rls = [compute_links(*e) for e in experiments_]
    rl = fusion_s_curve_score(rls, s_curve)
    return rl, rls


def linking_evaluation(rl, rls, Y):
    print("AP RPrec HPrec (Used for overall)")
    for rl_ in rls:
        print(*evaluate_linking(rl_, Y))
    print("AP RPrec HPrec (Overall)")
    print(*evaluate_linking(rl, Y))


def main():
    print("\n -- Training on st-jean -- ")
    _, _, _, X, Y = st_jean.parse_A()
    rl, rls = linking(X)
    linking_evaluation(rl, rls, Y)

    print(" -- Learning cut --")
    X_rl = rank_list_feature_extraction(rl)
    Y_rl = [1 if Y[a] == Y[b] else 0 for (a, b), score in rl]
    clf = LogisticRegression(random_state=0).fit(X_rl, Y_rl)
    print(clf.coef_)
    Y_pred = clf.predict(X_rl)
    print(precision_recall_fscore_support(Y_pred, Y_rl))

    ac = clustering(rl, clf)
    clustering_eval(Y, ac.labels_)

    print("\n -- Testing on st-jean B -- ")
    _, _, _, X_B, Y_B = st_jean.parse_B()
    rl, rls = linking(X_B)
    linking_evaluation(rl, rls, Y_B)
    ac = clustering(rl, clf)
    clustering_eval(Y_B, ac.labels_)

    print("\n -- Testing on brunet -- ")
    _, _, X, Y = brunet.parse()
    rl, rls = linking(X)
    linking_evaluation(rl, rls, Y)
    ac = clustering(rl, clf)
    clustering_eval(Y, ac.labels_)

    print("\n -- Testing on oxquarry -- ")
    _, X, Y = oxquarry.parse()
    rl, rls = linking(X)
    linking_evaluation(rl, rls, Y)
    ac = clustering(rl, clf)
    clustering_eval(Y, ac.labels_)


if __name__ == '__main__':
    main()
