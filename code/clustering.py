"""Clustering module."""

import numpy as np

from collections import defaultdict

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score

from rank_list_fusion import fusion_s_curve_score
from rank_list_fusion import fusion_z_score

from misc import distances_matrix_from_rank_list


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
    # n_clusters = compute_number_clusters_from_rank_list(rank_list[:np.sum(Y_pred)])

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


def unsupervised_clustering(rank_list, return_scores=False):
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    silhouette_by_clusters = {}

    ns = list(range(2, distances_matrix.shape[0]))

    for n in ns:
        args = {
            "n_clusters": n,
            "affinity": "precomputed",
            "linkage": "average",
        }
        ac = AgglomerativeClustering(**args)
        ac.fit(distances_matrix)

        silhouettes = silhouette_samples(distances_matrix, ac.labels_)
        score = np.median(silhouettes)

        if score > 0:
            silhouette_by_clusters[n] = (score, ac.labels_)
        else:
            break

    # use the closest to 0 silhouette score
    # in case the score never goes below 0
    silhouette_by_scores = dict(silhouette_by_clusters.values())
    minimal_score = min(np.abs(np.array(list(silhouette_by_scores.keys()))))
    best_labels = silhouette_by_scores[minimal_score]

    outputs = [best_labels]

    if return_scores:
        ns = silhouette_by_clusters.keys()
        scores = [silhouette_by_clusters[n][0] for n in ns]
        outputs += [(ns, scores)]

    return tuple(outputs)


def clustering_at_every_n_clusters(rank_list):
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    labels_list = []
    ns = list(range(2, distances_matrix.shape[0]))
    for n in ns:
        args = {
            "n_clusters": n,
            "affinity": "precomputed",
            "linkage": "average",
        }
        ac = AgglomerativeClustering(**args)
        ac.fit(distances_matrix)
        labels_list.append(ac.labels_)
    return (ns, labels_list)