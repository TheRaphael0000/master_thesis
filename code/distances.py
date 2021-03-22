"""Distance module."""

import math
import numpy as np


def manhattan(A, B):
    s = sum([abs(ai - bi) for ai, bi in zip(A, B)])
    return s


def tanimoto(A, B):
    s = sum([max(ai, bi) for ai, bi in zip(A, B)])
    return manhattan(A, B) / s


def matusita(A, B):
    s = sum([(np.sqrt(ai) - np.sqrt(bi))**2 for ai, bi in zip(A, B)])
    return np.sqrt(s)


def euclidean(A, B):
    s = sum([(ai - bi)**2 for ai, bi in zip(A, B)])
    return s ** 0.5


def clark(A, B):
    sum = 0
    for ai, bi in zip(A, B):
        try:
            v = (abs(ai - bi) / (ai + bi))**2
        except ZeroDivisionError:
            v = 0
        sum += v
    return sum ** 0.5


def cosine_sim(A, B):
    s1 = sum([ai * bi for ai, bi in zip(A, B)])
    s2 = sum([ai**2 for ai in A])
    s3 = sum([bi**2 for bi in B])
    d = (s2**0.5 * s3**0.5)
    if d == 0:
        d = 1e-10
    return s1 / d


def cosine(A, B):
    return math.acos(cosine_sim(A, B)) / math.pi


def j_divergence(A, B):
    B = [bi if bi != 0 else 1e-10 for bi in B]
    s = sum([(ai - bi) * math.log(ai / bi) for ai, bi in zip(A, B)])
    return s


if __name__ == '__main__':
    A = [1, 4, 9, 16]
    B = [0, 0, 0, 0]
    assert manhattan(A, B) == 30
    assert tanimoto(A, B) == 1
    assert matusita(A, B) == 30 ** 0.5
    assert clark(A, B) == 2
    # cosine_sim(A, B)
    # cosine(A, B)
    # j_divergence(A, B)
