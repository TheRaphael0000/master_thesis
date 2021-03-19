import matplotlib.pyplot as plt
import itertools
import numpy as np
from collections import Counter


def compute_r(Y):
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def dataset_infos(x, y):
    # #Texts, #Authors, Mean length, #Links
    return len(y), len(set(y)), round(np.mean([len(xi) for xi in x])), compute_r(y)


def zipf_law(total, n=21):
    distribution = dict(Counter(total.values()))
    keys = distribution.keys()
    A = list(range(1, n))
    B = [distribution[i] if i in distribution else 0 for i in A]
    plt.figure(figsize=(4, 3), dpi=200)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(A, B)
    plt.tight_layout()
    plt.savefig("zipf.png")


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
