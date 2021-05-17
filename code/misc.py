import itertools
from collections import Counter

import unicodedata
import numpy as np
from scipy.stats import binom_test
from scipy.stats import wilcoxon
from scipy.stats import weightedtau
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
    mean_token_length = round(np.mean(
        list(itertools.chain.from_iterable([[len(w) for w in xi] for xi in x]))), 3)
    return authors, texts, r, true_links, links, true_links_ratio, mean_length, mean_token_length


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
    plt.savefig("img/zipf.png")


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


def features_from_rank_list(rank_list):
    return [[np.log((i + 1) / len(rank_list)), score]
            for i, (link, score) in enumerate(rank_list)]


def labels_from_rank_list(rank_list, Y):
    return [1 if Y[a] == Y[b] else 0 for (a, b), score in rank_list]


def division(A, B):
    """Shorthand for the numpy division replacing division by 0 with 0"""
    return np.divide(A, B, where=B != 0, out=np.zeros(A.shape))


def log(A):
    """Shorthand for the numpy logarithm replacing 0 or negative log by 0"""
    return np.log(A, where=A > 0, out=np.full(A.shape, np.inf))


def normalize(x, a, b):
    """Normalize between a-b an array like"""
    input_range = np.max(x) - np.min(x)
    output_range = b - a
    return (x - np.min(x)) / input_range * output_range + a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_r(x):
    return - np.log((1 - x) / x)


def sign_test(A, B):
    pos = np.sum(A > B, axis=0)
    eq = np.sum(A == B, axis=0)
    neg = np.sum(A < B, axis=0)

    binom_tests = [binom_test([pos[i], neg[i]]) for i in range(pos.shape[0])]
    binom_tests = np.array(binom_tests)

    return np.array([pos, eq, neg]).T.tolist(), list(binom_tests)


def print_side_by_side_clusters(Y_true, Y_pred):
    for a in sorted(zip(Y_true, Y_pred), key=lambda x: x[-1]):
        print(a)


def rank_list_to_txt(rank_list, Y):
    with open("rank_list.txt", "w") as f:
        for rank, ((a, b), score) in enumerate(rank_list):
            f.write(f"{rank}, {'1' if Y[a] == Y[b] else '0'}, {str(score)}\n")


def normalize_text(s):
    # Removing accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    # To lowercase
    s = s.lower()
    return s


def first_letters_cut(X, n):
    return [[x[:n] for x in Xi] for Xi in X]


def last_letters_cut(X, n):
    return [[x[-n:] for x in Xi] for Xi in X]


def create_word_n_gram(word, n):
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def word_n_grams(X, n):
    texts = []
    for Xi in X:
        words = []
        for x in Xi:
            n_grams = create_word_n_gram(x, n)
            words.extend(n_grams)
        texts.append(words)
    return texts


def rank_list_distance(A, B):
    def rl_to_rank_dict(rl):
        return {link: i for i, (link, score) in enumerate(rl)}

    dA = rl_to_rank_dict(A)
    dB = rl_to_rank_dict(B)
    assert set(dA.keys()) == set(dB.keys())
    keys = dA.keys()
    
    vA, vB = [dA[k] for k in keys], [dB[k] for k in keys]
    correlation, pvalue = weightedtau(vA, vB)
    return correlation
