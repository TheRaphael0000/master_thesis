from collections import Counter
import itertools
import numpy as np
from statistics import mean


def rank_list_from_distances_matrix(distances_matrix):
    rank_list = []
    for a, b in itertools.combinations(range(distances_matrix.shape[0]), 2):
        link = a, b
        dist = distances_matrix[link]
        rank_list.append((link, dist))
    rank_list.sort(key=lambda x: x[-1])
    return rank_list


def distances_matrix_from_rank_list(rank_list):
    indices = list(itertools.chain.from_iterable([i[0] for i in rank_list]))
    max_ = max(indices) + 1
    distances_matrix = np.zeros((max_, max_,))
    for (a, b), dist in rank_list:
        distances_matrix[a, b] = dist
        distances_matrix[b, a] = dist
    return distances_matrix


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


def compute_r(Y):
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


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


def dataset_infos(x, y):
    # #Texts, #Authors, Mean length, #Links
    return len(y), len(set(y)), round(mean([len(xi) for xi in x])), compute_r(y)
