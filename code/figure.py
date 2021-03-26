import matplotlib.pyplot as plt
import numpy as np
from misc import sigmoid, sigmoid_reciprocal

def sigmoids():
    x = np.linspace(-8, 8, 100)
    y = sigmoid(x)
    plt.figure(figsize=(4,3), dpi=200)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("$S(x)$")
    plt.tight_layout()
    plt.savefig("sigmoid.png")

    x = np.linspace(sigmoid(-8), sigmoid(8), 100)
    y = sigmoid_reciprocal(x)
    plt.figure(figsize=(4,3), dpi=200)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("$S^{-1}(x)$")
    plt.tight_layout()
    plt.savefig("sigmoid_reciprocal.png")


if __name__ == '__main__':
    sigmoids()
