"""Distance module."""

import math
import numpy as np
from misc import division, log
from scipy.spatial.distance import euclidean as sp_euclidean
from scipy.spatial.distance import cityblock as sp_manhanttan


def manhattan(A, B):
    return sp_manhanttan(A, B)


def tanimoto(A, B):
    return manhattan(A, B) / np.sum(np.maximum(A, B))


def euclidean(A, B):
    return sp_euclidean(A, B)


def matusita(A, B):
    A = np.sqrt(A)
    B = np.sqrt(B)
    return euclidean(A, B)


def clark(A, B):
    A = np.array(A)
    B = np.array(B)
    abs = np.abs(A - B)
    sum = A + B
    s = np.sqrt(np.sum((division(abs, sum))**2))
    return s


def cosine_sim(A, B):
    s1 = np.dot(A, B)
    s2 = np.sqrt(np.dot(A, A))
    s3 = np.sqrt(np.dot(B, B))
    try:
        d = s1 / (s2 * s3)
    except ZeroDivisionError:
        d = 0
    return d


def cosine_distance(A, B):
    return 1 - cosine_sim(A, B)


def angular_distance(A, B):
    return np.arccos(cosine_sim(A, B)) / math.pi


def kld(A, B):
    return np.sum(A * log(division(A, B)))


def j_divergence(A, B):
    A = np.array(A)
    B = np.array(B)
    d = division(A, B)
    l = log(d)
    s = np.sum((A - B) * l)
    return s
