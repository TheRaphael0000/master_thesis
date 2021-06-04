"""Clustering module."""

import numpy as np

from collections import defaultdict

from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score

from misc import distances_matrix_from_rank_list
from misc import features_from_rank_list
from misc import labels_from_rank_list
from misc import fit_beta
from misc import find_two_beta_same_area


def agglomerative_clustering(rank_list, linkage="average", distance_threshold=np.inf):
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


def clustering_at_dist_thresh(rank_list, linkage="average", distance_threshold=np.inf):
    ac = agglomerative_clustering(rank_list, linkage, distance_threshold)
    for labels, dt in ac:
        pass
    return labels


from sklearn.cluster import AgglomerativeClustering

def unsupervised_clustering_old(rank_list, return_scores=False):
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

        silhouettes = silhouette_samples(distances_matrix, ac.labels_, metric="precomputed")
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


def unsupervised_clustering(rank_list, linkage="average", ips_stop=0):
    ac = agglomerative_clustering(rank_list, linkage, np.inf)
    distances_matrix = distances_matrix_from_rank_list(rank_list)

    labels, dts = zip(*(list(ac)))
    labels = labels[::-1][1:]

    mss = []
    current_labels = []
        

    for label in labels:
        nclusters = len(np.unique(label))
        
        ss = silhouette_samples(distances_matrix, label)
        ms = np.median(ss)
        print(nclusters, ms)
        current_labels.append(label)
        mss.append(ms)

        if ms <= ips_stop:
            n = [len(np.unique(cl)) for cl in current_labels]
            return label, (n, mss)


def clustering_at_every_n_clusters(rank_list, linkage="average"):
    ac = agglomerative_clustering(rank_list, linkage, np.inf)
    labels, dts = zip(*list(ac))
    n = [len(np.unique(label)) for label in labels]
    return (n, labels)
