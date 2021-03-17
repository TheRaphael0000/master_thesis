from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import math


def norm(x):
    return (x - min(x)) / (max(x) - min(x))


def sigmoid(n, alpha=10):
    def f(x):
        return 1 / (1 + math.exp(-x))
    y = np.linspace(-alpha, alpha, n)
    x = np.array([f(yi) for yi in y])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def tanh(n, alpha=10):
    def f(x):
        return np.tanh(x)
    y = np.linspace(-alpha, alpha, n)
    x = np.array([f(yi) for yi in y])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


def arctan(n, alpha=10):
    def f(x):
        return np.arctan(x)
    y = np.linspace(-alpha, alpha, n)
    x = np.array([f(yi) for yi in y])
    # normalize
    x, y = norm(x), norm(y)
    return x, y


if __name__ == '__main__':
    scale = 1000
    plt.figure(figsize=(4, 3), dpi=200)

    x1, y1 = sigmoid(scale)
    plt.plot(x1, y1, label="sigmoid")

    x2, y2 = tanh(scale)
    plt.plot(x2, y2, label="tanh")

    x3, y3 = arctan(scale)
    plt.plot(x3, y3, label="arctan")

    plt.legend()
    plt.tight_layout()
    plt.savefig("s_curve.png")

    assert x1[0] == 0.0 and x1[-1] == 1.0
    assert y1[0] == 0.0 and y1[-1] == 1.0
    assert x2[0] == 0.0 and x2[-1] == 1.0
    assert y2[0] == 0.0 and y2[-1] == 1.0
    assert x3[0] == 0.0 and x3[-1] == 1.0
    assert y3[0] == 0.0 and y3[-1] == 1.0
