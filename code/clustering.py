"""Clustering module."""

import numpy as np

from collections import defaultdict

from scipy.stats import beta
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score

from misc import distances_matrix_from_rank_list
from misc import features_from_rank_list
from misc import labels_from_rank_list
from misc import fit_beta
from misc import find_two_beta_same_area


def dist_thresh_logistic_regression(rl_training, Y_training, rl_testing, prob_threshold=0.5):
    X_rl = features_from_rank_list(rl_training)
    Y_rl = labels_from_rank_list(rl_training, Y_training)
    model = LogisticRegression(random_state=0).fit(X_rl, Y_rl)
    
    # compute distance threshold
    X = features_from_rank_list(rl_testing)
    # fit the model
    probs = model.predict_proba(X)[:,1]
        
    if np.sum(probs == prob_threshold) >= 1:
        # exact probability threshold
        pos = np.sum(probas >= prob_threshold)
        dt = rl_testing[pos][-1]
        return dt
    else:
        # linear interpolation of the probability
        i_s = list(range(len(probs)))
        pos1 = i_s[np.sum(probs > prob_threshold)]
        pos2 = i_s[::-1][np.sum(probs < prob_threshold)]
        prob1 = probs[pos1]
        prob2 = probs[pos2]
        dt1 = rl_testing[pos1][-1]
        dt2 = rl_testing[pos2][-1]

        alpha = (prob_threshold - prob1) / (prob2 - prob1)
        dt = (dt2 - dt1) * alpha + dt1
        return dt


def dist_thresh_two_beta(rl_training, Y_training):
    links = np.array(list(zip(*rl_training))[0])
    scores = np.array(list(zip(*rl_training))[1])
    labels = np.array([Y_training[a] == Y_training[b] for a, b in links])
    
    # if it's not a probability normalize between 0 and 1
    normalize = False
    if np.max(scores) > 1 or np.min(scores) < 0:
        normalize = True
        min_ = np.min(scores)
        input_range = np.max(scores) - min_
        scores = (scores - min_) / input_range

    beta_true = fit_beta(scores[labels])
    beta_false = fit_beta(scores[~labels])
        
    dt = find_two_beta_same_area(beta_true, beta_false)
    
    # un-normalize
    if normalize:
        dt = dt * input_range + min_
        
    return dt


def clustering_at_dist_thresh(distance_threshold, rank_list, linkage="average"):
    args = {
        "n_clusters": None,
        "affinity": "precomputed",
        "linkage": linkage,
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
