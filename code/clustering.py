"""Clustering module

This module contains fonction relative to the clustering task.
The clustering function require rank lists which are generated using for example function from the linking module.
"""

from collections import defaultdict

from misc import distances_matrix_from_rank_list
from misc import features_from_rank_list
from misc import fit_beta
from misc import find_two_beta_same_area
from misc import labels_from_rank_list
from evaluate import evaluate_clustering

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score


def agglomerative_clustering(rank_list, linkage="average", distance_threshold=np.inf):
    """Agglomerative clustering algorithm

    Keyword arguments:
        rank_list -- Rank list to used to base the clustering
        linkage -- Linkage criterion to use at each merge (default: average)
        distance_threshold -- Distance threshold at which the algorithm must stop (optional)

    Return:
        generator -- Generator over the algorithm steps

    The generator yield at each step:
        - The clusters
        - The last distance used for the merge

    If the minimal distance is greater than the distance_threshold, the generator stop at this step.
    Otherwise the generator stop when it reaches 1 cluster.
    """
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    np.fill_diagonal(distances_matrix, np.inf)
    w, _ = distances_matrix.shape
    groups = [[i] for i in range(w)]

    n = np.inf
    i = 0

    linkages = {
        "single": np.min,
        "complete": np.max,
        "average": np.average
    }
    linkage_func = linkages[linkage]

    yield np.arange(0, w, dtype=int), -np.inf

    while n > 1:
        fmatrix = distances_matrix.flatten()
        p = np.argmin(fmatrix)

        _, ch = distances_matrix.shape
        y = p // ch
        x = p - ch * y
        current_value = distances_matrix[y, x]

        if current_value > distance_threshold:
            break

        merge = [groups[x] + groups[y]]
        del groups[x]
        del groups[y]
        groups += merge

        row = np.array([distances_matrix[x, :], distances_matrix[y, :]])
        col = np.array([distances_matrix[:, x], distances_matrix[:, y]])
        row = linkage_func(row, axis=0)
        col = linkage_func(col, axis=0)
        col = np.delete(col, [x, y])
        row = np.delete(row, [x, y])

        distances_matrix = np.delete(distances_matrix, [x, y], 0)
        distances_matrix = np.delete(distances_matrix, [x, y], 1)

        distances_matrix = np.vstack((distances_matrix, row))
        col = np.hstack((col, np.array([np.inf])))
        distances_matrix = np.column_stack((distances_matrix, col))

        # create labels based on groups
        labels = np.empty([w], dtype=int)

        for i, group in enumerate(groups):
            for v in group:
                labels[v] = i

        yield labels, current_value

        n = len(np.unique(labels))


def dist_thresh_logistic_regression(rl_training, Y_training, rl_testing, prob_threshold=0.5):
    """Find a distance threshold using the clustering logistic regression-based algorithm.

    Keyword arguments:
        rl_training -- Rank list to use to train the logistic regression model
        Y_training -- Labels for each document
        rl_testing -- Rank list to fit the model on
        prob_threshold -- Threshold to split the true from the false links

    Return:
        float -- the distance threshold value
    """
    X_rl = features_from_rank_list(rl_training)
    Y_rl = labels_from_rank_list(rl_training, Y_training)
    model = LogisticRegression(random_state=0).fit(X_rl, Y_rl)

    # compute distance threshold
    X = features_from_rank_list(rl_testing)
    # fit the model
    probs = model.predict_proba(X)[:, 1]

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
    """Find a distance threshold using the two beta-based algorithm.

    Keyword arguments:
        rl_training -- Rank list to use to find the true and false link distribution
        Y_training -- Labels for each document

    Return:
        float -- the distance threshold value
    """
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


def clustering_at_dist_thresh(rank_list, linkage="average", distance_threshold=np.inf):
    """Simple function that execute the clustering algorithm entierly and return the final labels.
    This is useful when the distance threshold is known (supervised methods).
    """
    ac = agglomerative_clustering(rank_list, linkage, distance_threshold)
    for labels, dt in ac:
        pass
    return labels


def silhouette_based_clustering(rank_list, linkage="average", alpha=0):
    """Find a clustering using the Silhouette-based algorithm.

    Keyword arguments:
        rank_list -- Rank list to use on the hirachical clustering.
        linkage -- Linkage criterion (default: average)
        alpha -- The proportion to remove from the maximal mean Silhouette score. The sign gives the direction (negative: left, positive: right), a good value for authorship clustering is around 0.2 (default: 0)

    Return:
        labels_ -- The labels obtained at the end of the algorithm
        (n, sss) -- A tuple of lists, they draw the (number clusters),(mean silhouette score) graph.
    """
    ac = agglomerative_clustering(rank_list, linkage, np.inf)
    distances_matrix = distances_matrix_from_rank_list(rank_list)

    # normalize the distance matrix for the silhouette computation
    min_ = np.nanmin(distances_matrix)
    max_ = np.nanmax(distances_matrix)
    distances_matrix = (distances_matrix - min_) / (max_ - min_)
    np.fill_diagonal(distances_matrix, 0)

    labels, dts = zip(*(list(ac)))
    labels = labels[1:-1][::-1]

    sss = []
    n = []

    for labels_ in labels:
        ss = silhouette_score(distances_matrix, labels_, metric="precomputed")
        sss.append(ss)
        n.append(len(np.unique(labels_)))

    max_i = np.argmax(sss)

    left_side, right_side = labels[:max_i], labels[max_i + 1:]
    left_scores, right_scores = sss[:max_i], sss[max_i + 1:]
    max_score = sss[max_i]

    target = max_score - np.abs(alpha) * max_score

    if alpha < 0:
        scores = np.abs(target - left_scores)
        target_i = np.argmin(scores)
        labels_ = labels[target_i]
    elif alpha > 0:
        scores = np.abs(target - right_scores)
        target_i = np.argmin(scores)
        labels_ = labels[target_i + max_i + 1]
    else:
        labels_ = labels[max_i]

    return labels_, (n, sss)


def clustering_at_every_n_clusters(rank_list, linkage="average"):
    """Run the hierarchical clustering algorithm at each number of clusters

    Keyword arguments:
        rank_list -- Rank list to use on the hirachical clustering.
        linkage -- Linkage criterion (default: average)

    Return:
        (n, labels) -- A tupple, the first element is a list which correspond to each number of clusters, the second element is a list of labels obtained at the end of the agglomerative clustering at each number of clusters.
    """
    ac = agglomerative_clustering(rank_list, linkage, np.inf)
    labels, dts = zip(*list(ac))
    n = [len(np.unique(label)) for label in labels]
    return (n, labels)


def best_clustering(rank_list, linkage, Y):
    """Run the hierarchical clustering algorithm at each step. Return the B^3_{F_1} for each step.

    Keyword arguments:
        rank_list -- Rank list to use on the hirachical clustering.
        linkage -- Linkage criterion
        Y -- The labels for the documents in the rank list

    Return:
        float -- The B^3_{F_1}
    """
    ac = agglomerative_clustering(rank_list, linkage, np.inf)
    best = None
    for labels, dt in ac:
        m = evaluate_clustering(Y, labels)
        if best is None or m[0] > best[0]:
            best = m
    return best[0]
