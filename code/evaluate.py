"""Evaluation module

This module contains function to the evaluate either Rank lists or Clustering results.
The rank list are evaluated using information retrieval techniques : AP (Average precision), R-Prec (R-Precision) and H-Prec (High-Precision)
The clustering results are evaluated using the B^3 metrics family and a metric called r-ratio-diff (r-diff)
"""

from misc import compute_r

import bcubed
import numpy as np


def evaluate_linking(rank_list, Y):
    """Combination of metrics used for evaluating the linking task

    Keyword arguments:
        rank_list -- Rank list to evaluate
        Y -- The labels for the documents in the rank list

    Return:
        np.array -- The array contains the AP, RPrec, HPrec for this rank list
    """
    ap_ = ap(rank_list, Y)
    rprec_ = rprec(rank_list, Y)
    hprec_ = hprec(rank_list, Y)
    M = np.array([ap_, rprec_, hprec_])
    return M


def evaluate_clustering(Y_true, Y_pred):
    """Combination of metrics used for evaluating the clustering task

    Keyword arguments:
        Y_true -- The true labels, groundtruth
        Y_pred -- The predicted labels, the clustering result

    Return:
        np.array -- The array contains : B^3 f1-score, B^3 precision, B^3 recall and the r-ratio-diff for this clustering result
    """
    # change clustering representation for bcubed computations
    ldict = {}
    cdict = {}
    for i, (l, c) in enumerate(zip(Y_true, Y_pred)):
        ldict[i] = set([l])
        cdict[i] = set([c])
    bcubed_precision = bcubed.precision(cdict, ldict)
    bcubed_recall = bcubed.recall(cdict, ldict)
    bcubed_fscore = bcubed.fscore(bcubed_precision, bcubed_recall)
    r_ratios_diff_ = r_ratios_diff(Y_true, Y_pred)
    M = np.array([bcubed_fscore, bcubed_precision,
                  bcubed_recall, r_ratios_diff_])
    return M


def precision_at_k(rank_list, Y, k):
    """Evaluate the precision at K in a rank list

    Keyword arguments:
        rank_list -- The rank list to evaluate
        Y -- The labels for the documents in the rank list
        k -- The k, correspond to the rank where the precision mean is stopped

    Return:
        float -- The precision at k for this rank list
    """
    sum = 0
    for link, distance in rank_list[0:k]:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            sum += 1
    return sum / k


def ap(rank_list, Y):
    """Compute the average precision on a rank list

    Keyword arguments:
        rank_list -- The rank list to evaluate
        Y -- The labels for the documents in the rank list

    Return:
        float -- The average precision for this rank list
    """
    sum_precisions = 0
    correct = 0
    for i, (link, distance) in enumerate(rank_list):
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            correct += 1
            precision_at_i = correct / (i + 1)
            sum_precisions += precision_at_i
    return sum_precisions / correct


def rprec(rank_list, Y):
    """Find the number of links in a dataset than compute the precision@r

    Keyword arguments:
        rank_list -- The rank list to evaluate
        Y -- The labels for the documents in the rank list

    Return:
        float -- The R-Precision for this rank list
    """
    R = compute_r(Y)
    return precision_at_k(rank_list, Y, R)


def hprec(rank_list, Y):
    """Find the number of consecutive true links in a rank list

    Keyword arguments:
        rank_list -- The rank list to evaluate
        Y -- The labels for the documents in the rank list

    Return:
        int -- The HPrec for this rank list
    """
    for i, (link, distance) in enumerate(rank_list):
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        # once the first incorrect sample is found
        # we can return the iterator position
        if ya != yb:
            return i
    # every documents in the list are relevant at this point
    return len(rank_list)


def r_ratios_diff(Y_true, Y_pred):
    """Compute the difference in r-ratio between two clusterings

    Keyword arguments:
        Y_true -- The true labels, groundtruth
        Y_pred -- The predicted labels, the clustering result

    Return:
        float -- The r_ratios_diff, also called r-diff
    """
    k = len(np.unique(Y_true))
    p = len(np.unique(Y_pred))
    N = len(Y_true)
    return (p - k) / N
