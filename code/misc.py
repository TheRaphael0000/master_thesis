import itertools
from collections import Counter

import numpy as np
from scipy.stats import binom_test
import matplotlib.pyplot as plt


def compute_r(Y):
    """Find the number of true links in a dataset"""
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def dataset_infos(x, y):
    """Different basic metrics used to evaluate a dataset"""
    authors = len(set(y))
    texts = len(y)
    r = round(authors / texts, 3)
    true_links = compute_r(y)
    links = (texts * (texts - 1)) // 2
    true_links_ratio = round(true_links / links, 3)
    mean_length = round(np.mean([len(xi) for xi in x]))
    return authors, texts, r, true_links, links, true_links_ratio, mean_length


def zipf_law(total, n=21):
    """Plot the zipf law of a dictionary of words frequencies"""
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
    """Create a rank list from a distance matrix"""
    rank_list = []
    for a, b in itertools.combinations(range(distances_matrix.shape[0]), 2):
        link = a, b
        dist = distances_matrix[link]
        rank_list.append((link, dist))
    rank_list.sort(key=lambda x: x[-1])
    return rank_list


def distances_matrix_from_rank_list(rank_list):
    """Create a distance matrix from a rank list"""
    indices = list(itertools.chain.from_iterable([i[0] for i in rank_list]))
    max_ = max(indices) + 1
    distances_matrix = np.zeros((max_, max_,))
    for (a, b), dist in rank_list:
        distances_matrix[a, b] = dist
        distances_matrix[b, a] = dist
    return distances_matrix


def division(A, B):
    """Shorthand for the numpy division replacing division by 0 with 0"""
    return np.divide(A, B, where=B != 0, out=np.zeros(A.shape))


def log(A):
    """Shorthand for the numpy logarithm replacing 0 or negative log by 0"""
    return np.log(A, where=A > 0, out=np.zeros(A.shape))


def normalize(x):
    """Normalize between 0-1 an array like"""
    num = x - np.min(x)
    div = np.max(x) - np.min(x)
    if div == 0:
        raise Exception("can't normalize")
    return num / div


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_r(x):
    return - np.log((1 - x) / x)


def sign_test(A, B):
    pos = np.sum(A > B, axis=0)
    eq = np.sum(A == B, axis=0)
    neg = np.sum(A < B, axis=0)

    print(pos)
    print(eq)
    print(neg)

    binom_tests = [binom_test([pos[i], neg[i]]) for i in range(pos.shape[0])]
    binom_tests = np.array(binom_tests)
    print(binom_tests)


def rank_list_to_txt(rank_list, Y):
    with open("rank_list.txt", "w") as f:
        for rank, ((a,b), score) in enumerate(rank_list):
            f.write(f"{rank}, {'1' if Y[a] == Y[b] else '0'}, {str(score)}\n")
