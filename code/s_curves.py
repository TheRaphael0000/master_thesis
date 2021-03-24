"""S-curves module"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math


def norm(x):
    return (x - min(x)) / (max(x) - min(x))


def linear(n):
    x = np.linspace(0, 1.0, n)
    y = x
    return x, y


def sigmoid(n, zoom_factor=3):
    alpha = 2 ** zoom_factor

    def f(x):
        return 1 / (1 + math.exp(-x))

    def f_r(x):
        return np.log((1 - x) / x)
    x = np.linspace(f(-alpha), f(alpha), n)
    y = np.array([-f_r(xi) for xi in x])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def tanh(n, zoom_factor=3):
    alpha = 2 ** zoom_factor

    def f(x):
        return np.tanh(x)

    def f_r(x):
        return np.arctanh(x)
    x = np.linspace(f(-alpha), f(alpha), n)
    y = np.array([f_r(xi) for xi in x])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def arctan(n, zoom_factor=2**3):
    alpha = 2 ** zoom_factor

    def f(x):
        return np.arctan(x)

    def f_r(x):
        return np.tan(x)
    x = np.linspace(f(-alpha), f(alpha), n)
    y = np.array([f_r(xi) for xi in x])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


if __name__ == '__main__':
    scale = 500
    plt.figure(figsize=(4, 3), dpi=200)

    min_ = 0
    max_ = 4
    zoom_factors = np.arange(min_, max_, 0.06)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.hsv(np.linspace(0, 1, len(zoom_factors))))

    for i in zoom_factors:
        x, y = tanh(scale, i)
        plt.plot(x, y, linewidth=0.2)

    plt.colorbar(plt.cm.ScalarMappable(
        norm=colors.Normalize(min_, max_), cmap="hsv"))
    # plt.legend()
    plt.tight_layout()
    plt.savefig("s_curves.png")
