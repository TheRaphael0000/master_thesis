"""Evaluation module."""

from misc import compute_r

import numpy as np

import bcubed


def evaluate_linking(rank_list, Y):
    """Combination of metrics used for evaluating the linking task"""
    ap_ = ap(rank_list, Y)
    rprec_ = rprec(rank_list, Y)
    hprec_ = hprec(rank_list, Y)
    M = np.array([ap_, rprec_, hprec_])
    return M


def evaluate_clustering(Y_true, Y_pred):
    """Combination of metrics used for evaluating the clustering task"""
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
    M = np.array([bcubed_fscore, bcubed_precision, bcubed_recall, r_ratios_diff_])
    return M


def precision_at_k(rank_list, Y, k):
    """Evaluate the precision at K in a rank list"""
    sum = 0
    for link, distance in rank_list[0:k]:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            sum += 1
    return sum / k


def ap(rank_list, Y):
    """Compute the average precision on a rank list"""
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
    """Find the number of links in a dataset than compute the precision@r"""
    R = compute_r(Y)
    return precision_at_k(rank_list, Y, R)


def hprec(rank_list, Y):
    """Find the number of first correct links in the rank list"""
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
    k = len(np.unique(Y_true))
    p = len(np.unique(Y_pred))
    N = len(Y_true)
    return (p - k) / N
