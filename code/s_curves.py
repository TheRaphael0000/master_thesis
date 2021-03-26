"""S-curves module"""

import matplotlib.pyplot as plt
import numpy as np

from misc import normalize, sigmoid, sigmoid_reciprocal


def linear():
    """Return a linear function with n points"""
    def generator(n):
        x = np.linspace(0, 1.0, n)
        y = x
        return x, y
    return generator


def sigmoid_reciprocal(c=4, r=0.25):
    """Return a sigmoid reciprocal function with n points,
    the zoom factor is based on the sigmoid function"""
    def generator(n):
        alpha = c
        n1 = int(n * r)
        n2 = int(n - n1)
        x1 = np.linspace(sigmoid(-alpha), sigmoid(0), n1, endpoint=False)
        x2 = np.linspace(sigmoid(0), 1 - sigmoid(-alpha), n2)
        x = np.array(list(x1) + list(x2))
        y = np.array([sigmoid_reciprocal(xi) for xi in x])
        x = np.array(range(n))
        # normalize
        x, y = normalize(x), normalize(y)
        return x, y
    return generator


if __name__ == '__main__':
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(*sigmoid_reciprocal()(500))
    plt.tight_layout()
    plt.savefig("s_curves.png")
