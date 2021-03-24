import itertools
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt


def compute_r(Y):
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def dataset_infos(x, y):
    texts = len(y)
    authors = len(set(y))
    mean_length = round(np.mean([len(xi) for xi in x]))
    links = compute_r(y)
    r = round(authors / texts, 2)
    return texts, authors, mean_length, links, r


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


def division(A, B):
    return np.divide(A, B, where=B!=0, out=np.zeros(A.shape))


def log(A):
    return np.log(A, where=A>0, out=np.zeros(A.shape))
