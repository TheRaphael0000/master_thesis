"""S-curves module"""

import matplotlib.pyplot as plt
import numpy as np
import math


def norm(x):
    return (x - min(x)) / (max(x) - min(x))


def linear(n):
    x = np.linspace(0, 1.0, n)
    y = x
    return x, y


def sigmoid(n, alpha=8):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def f_r(x):
        return np.log((1 - x) / x)
    x = np.linspace(f(-alpha), f(alpha), n)
    y = np.array([-f_r(xi) for xi in x])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def tanh(n, alpha=8):
    def f(x):
        return np.tanh(x)

    def f_r(x):
        return np.arctanh(x)
    x = np.linspace(f(-alpha), f(alpha), n)
    y = np.array([f_r(xi) for xi in x])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def arctan(n, alpha=8):
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
    scale = 1000
    plt.figure(figsize=(4, 3), dpi=200)

    x1, y1 = linear(scale)
    plt.plot(x1, y1, label="linear")

    x2, y2 = sigmoid(scale)
    plt.plot(x2, y2, label="sigmoid")

    x3, y3 = tanh(scale)
    plt.plot(x3, y3, label="tanh")

    x4, y4 = arctan(scale)
    plt.plot(x4, y4, label="arctan")

    plt.legend()
    plt.tight_layout()
    plt.savefig("s_curves.png")

    assert x1[0] == 0.0 and x1[-1] == 1.0
    assert y1[0] == 0.0 and y1[-1] == 1.0
    assert x2[0] == 0.0 and x2[-1] == 1.0
    assert y2[0] == 0.0 and y2[-1] == 1.0
    assert x3[0] == 0.0 and x3[-1] == 1.0
    assert y3[0] == 0.0 and y3[-1] == 1.0
    assert x4[0] == 0.0 and x4[-1] == 1.0
    assert y4[0] == 0.0 and y4[-1] == 1.0
