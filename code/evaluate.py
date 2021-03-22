"""Evaluation module."""

from misc import compute_r

import bcubed
from sklearn.metrics.cluster import adjusted_mutual_info_score


def evaluate_linking(rank_list, Y):
    return ap(rank_list, Y), rprec(rank_list, Y), hprec(rank_list, Y)


def evaluate_clustering(Y, pred):
    # change clustering representation for bcubed computations
    ldict = {}
    cdict = {}
    for i, (l, c) in enumerate(zip(Y, pred)):
        ldict[i] = set([l])
        cdict[i] = set([c])
    bcubed_precision = bcubed.precision(cdict, ldict)
    bcubed_recall = bcubed.recall(cdict, ldict)
    bcubed_fscore = bcubed.fscore(bcubed_precision, bcubed_recall)
    mutual_info_score = adjusted_mutual_info_score(Y, pred)
    return bcubed_precision, bcubed_recall, bcubed_fscore, mutual_info_score


def precision_at_k(rank_list, Y, k):
    sum = 0
    for link, distance in rank_list[0:k]:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            sum += 1
    return sum / k


def ap(rank_list, Y):
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
    R = compute_r(Y)
    return precision_at_k(rank_list, Y, R)


def hprec(rank_list, Y):
    i = 0
    for link, distance in rank_list:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            i += 1
        else:
            break
    return i
