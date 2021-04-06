"""Clustering module."""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

import distances
import s_curves
from corpus import oxquarry
from corpus import brunet
from corpus import st_jean
from corpus import pan16
from rank_list_fusion import compute_multiple_links
from misc import dataset_infos
from misc import distances_matrix_from_rank_list
from misc import rank_list_to_txt
from evaluate import evaluate_linking
from evaluate import evaluate_clustering


def clustering(rank_list, clf):
    X = [[np.log(i+1), score] for i, (link, score) in enumerate(rank_list)]
    Y_pred = clf.predict(X)

    # the sum give the position n of the "flip" in the rank list since
    # the n first should be ones
    i = np.sum(Y_pred)
    ajusted_i = i#int(i * 1.4)
    distance_threshold = rank_list[ajusted_i][-1]
    print(i, ajusted_i, distance_threshold)

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


def clustering_case():
    _, _, _, X, Y = st_jean.parse()
    # _, _, X, Y = brunet.parse()

    n = len(X) // 2
    X_train = X[:n]
    Y_train = Y[:n]
    X_test = X[n:]
    Y_test = Y[n:]

    print("authors, texts, r, true_links, links, true_links_ratio, mean_length")
    print(*dataset_infos(X_train, Y_train))
    print(*dataset_infos(X_test, Y_test))

    print()
    print(" -- Training -- ")
    print()

    def experiments(Xi):
        return [
            [Xi, 0, 500, True, 1e-1, distances.manhattan],
            [Xi, 0, 500, False, 1e-1, distances.tanimoto],
            [Xi, 0, 500, False, 1e-1, distances.clark],
            [Xi, 0, 500, False, 1e-1, distances.matusita],
            [Xi, 0, 500, True, 1e-1, distances.cosine_distance],

            [Xi, 6, 500, True, 1e-1, distances.manhattan],
            [Xi, 6, 500, False, 1e-1, distances.tanimoto],
            # [Xi, 6, 500, False, 1e-1, distances.clark],
            [Xi, 6, 500, False, 1e-1, distances.matusita],
            # [Xi, 6, 500, False, 1e-1, distances.cosine_distance],
        ]
    s_curve = s_curves.sigmoid_reciprocal()

    print(" -- Linking -- ")
    experiments_train = experiments(X_train)
    rl_train, rls_train = compute_multiple_links(
        experiments_train, s_curve)

    print(" -- Linking evaluation -- ")
    print("AP RPrec HPrec (Used for overall)")
    for rl in rls_train:
        print(*evaluate_linking(rl, Y_train))
    print("AP RPrec HPrec (Overall)")
    print(*evaluate_linking(rl_train, Y_train))

    X_rl = [[np.log(i+1), score] for i, ((a, b), score) in enumerate(rl_train)]
    Y_rl = [1 if Y_train[a] == Y_train[b] else 0 for (a, b), score in rl_train]
    clf = LogisticRegression(random_state=0).fit(X_rl, Y_rl)

    print(" -- Clustering -- ")
    ac = clustering(rl_train, clf)

    print(" -- Clustering Evaluation -- ")
    b3_prec, b3_rec, b3_fscore, mis = evaluate_clustering(Y_train, ac.labels_)
    print("bcubed.precision", b3_prec)
    print("bcubed.recall", b3_rec)
    print("bcubed.fscore", b3_fscore)
    print("adjusted_mutual_info_score", mis)


    print()
    print(" -- Testing -- ")
    print()

    print(" -- Linking -- ")
    experiments_test = experiments(X_test)
    rl_test, rls_test = compute_multiple_links(
        experiments_test, s_curve)

    print(" -- Linking evaluation -- ")
    print("AP RPrec HPrec (Used for overall)")
    for rl in rls_test:
        print(*evaluate_linking(rl, Y_test))
    print("AP RPrec HPrec (Overall)")
    print(*evaluate_linking(rl_test, Y_test))

    print(" -- Clustering -- ")
    ac = clustering(rl_test, clf)

    # for a in sorted(zip(Y_test, ac.labels_), key=lambda x:x[-1]):
    #     print(a)

    print(" -- Clustering Evaluation -- ")
    b3_prec, b3_rec, b3_fscore, mis = evaluate_clustering(Y_test, ac.labels_)
    print("bcubed.precision", b3_prec)
    print("bcubed.recall", b3_rec)
    print("bcubed.fscore", b3_fscore)
    print("adjusted_mutual_info_score", mis)


if __name__ == '__main__':
    clustering_case()
