"""Clustering module."""

import numpy as np

from collections import defaultdict

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score

from misc import distances_matrix_from_rank_list
from misc import features_from_rank_list
from misc import labels_from_rank_list


def supervised_clustering_training(rank_list, Y, return_eval=False, random_state=0):
    X_rl = features_from_rank_list(rank_list)
    Y_rl = labels_from_rank_list(rank_list, Y)
    model = LogisticRegression(random_state=random_state).fit(X_rl, Y_rl)
    outputs = [model]
    if return_eval:
        Y_pred = model.predict(X_rl)
        eval = precision_recall_fscore_support(Y_pred, Y_rl)
        return (model, eval)
    else:
        return model


def supervised_clustering_predict(model, rank_list):
    # compute distance threshold
    X = features_from_rank_list(rank_list)
    pos = np.sum(model.predict(X))
    # both lines equaivalent
    # pos = np.sum(model.predict_proba(X)[:,1] > 0.5)

    distance_threshold = rank_list[pos][-1]

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


def unsupervised_clustering(rank_list, ips_stop=0, return_scores=False):
    distances_matrix = distances_matrix_from_rank_list(rank_list)

    ns = list(range(2, distances_matrix.shape[0]))
    silhouette_scores = []

    labels = np.zeros((distances_matrix.shape[0],), dtype=int)

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

        if score <= ips_stop:
            break

        labels = ac.labels_
        silhouette_scores.append(score)

    if not return_scores:
        return labels
    
    explored_ns = ns[:len(silhouette_scores)]
    scores = (explored_ns, silhouette_scores)

    return labels, scores


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
