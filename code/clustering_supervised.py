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


def supervised_clustering_feature_extraction(rank_list):
    X = [[np.log((i + 1) / len(rank_list)), score]
         for i, (link, score) in enumerate(rank_list)]
    return X


def supervised_clustering_training(rank_list, Y, return_eval=False, random_state=0):
    X_rl = supervised_clustering_feature_extraction(rank_list)
    Y_rl = [1 if Y[a] == Y[b] else 0 for (a, b), score in rank_list]
    model = LogisticRegression(random_state=random_state).fit(X_rl, Y_rl)
    outputs = [model]
    if return_eval:
        Y_pred = model.predict(X_rl)
        eval = precision_recall_fscore_support(Y_pred, Y_rl)
        outputs += [eval]
    return tuple(outputs)


def supervised_clustering_predict(model, rank_list):
    # compute distance threshold
    X = supervised_clustering_feature_extraction(rank_list)
    Y_pred = model.predict(X)

    # the sum give the position n of the "flip" in the rank list since
    # the n first should be ones
    distance_threshold = rank_list[np.sum(Y_pred)][-1]

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
    labels = ac.labels_
    return labels


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
    model, eval = supervised_clustering_training(rl, Y, return_eval=True)
    print(eval)

    labels = supervised_clustering_predict(model, rl)
    print(evaluate_clustering(Y, labels))

    print("\n -- Testing on st-jean B -- ")
    _, _, _, X_B, Y_B = st_jean.parse_B()
    rl, rls = linking(X_B)
    linking_evaluation(rl, rls, Y_B)
    labels = supervised_clustering_predict(model, rl)
    print(evaluate_clustering(Y_B, labels))

    print("\n -- Testing on brunet -- ")
    _, _, X, Y = brunet.parse()
    rl, rls = linking(X)
    linking_evaluation(rl, rls, Y)
    labels = supervised_clustering_predict(model, rl)
    print(evaluate_clustering(Y, labels))

    print("\n -- Testing on oxquarry -- ")
    _, X, Y = oxquarry.parse()
    rl, rls = linking(X)
    linking_evaluation(rl, rls, Y)
    labels = supervised_clustering_predict(model, rl)
    print(evaluate_clustering(Y, labels))


if __name__ == '__main__':
    main()
