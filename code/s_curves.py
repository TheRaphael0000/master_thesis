"""S-curves module"""

import matplotlib.pyplot as plt
import numpy as np
import math

from misc import normalize, sigmoid, sigmoid_r


def soft_veto(rank_list, s_curve):
    y = s_curve(len(rank_list))
    updated_rank_list = []
    for i, (link, score) in enumerate(rank_list):
        if y[i] == np.inf:
            new_score = np.inf
        elif y[i] == -np.inf:
            new_score = -np.inf
        else:
            new_score = score * y[i]
        updated_rank_list.append((link, new_score))
    return updated_rank_list


def sigmoid_reciprocal(c=4, r=0.25):
    """Return a sigmoid reciprocal function with n points,
    the zoom factor is based on the sigmoid function"""
    def generator(n):
        n1 = int(n * r)
        n2 = n - n1
        x1 = np.linspace(sigmoid(-c), sigmoid(0), n1, endpoint=False)
        x2 = np.linspace(sigmoid(0), 1 - sigmoid(-c), n2)
        y = np.array([sigmoid_r(xi) for xi in list(x1) + list(x2)])
        return y
    return generator


def full_boost(top, bottom):
    def generator(n):
        n1 = math.ceil(n * top)
        n2 = math.ceil(n * (1 - bottom))
        if n1 > n2:
            raise Exception("Overlapping veto")
        y = np.ones((n,))
        y[:n1] = -np.inf
        y[n2:] = np.inf
        return y
    return generator

if __name__ == '__main__':
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(*sigmoid_reciprocal()(500))
    plt.tight_layout()
    plt.savefig("img/s_curves.png")
