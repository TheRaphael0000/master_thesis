"""S-curves module

The S-curves are curves used for the veto and soft-veto.
These veto a used in the fusion step.
"""

import math

from misc import normalize
from misc import sigmoid
from misc import sigmoid_r

import matplotlib.pyplot as plt
import numpy as np


def soft_veto(rank_list, s_curve):
    """Apply a soft-veto curve to a rank list

    Arguments:
        rank_list -- The rank list to apply the soft-veto to
        s_curve -- The s_curve, correspond to the list of floats

    Returns:
        updated_rank_list -- The rank list with updated scores
    """
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
    the zoom factor is based on the sigmoid function

    Arguments:
        c -- The c parameter, influance the steepness of the sigmoid
        r -- The x position of the y center of the curve

    Returns:
        wrapper -- A wrapper for this function
    """
    def wrapper(n):
        """sigmoid_reciprocal wrapper, use the arguments from the parent function to create the curve

        Arguments:
            n -- The number of points to sample on the function

        Returns:
            y -- An array of size n, containing the values of the reciprocal sigmoid
        """
        n1 = int(n * r)
        n2 = n - n1
        x1 = np.linspace(sigmoid(-c), sigmoid(0), n1, endpoint=False)
        x2 = np.linspace(sigmoid(0), 1 - sigmoid(-c), n2)
        y = np.array([sigmoid_r(xi) for xi in list(x1) + list(x2)])
        return y
    return wrapper


def full_boost(top, bottom):
    """A full boost s-curve (not used)

    Arguments:
        top -- The top proportion to fully boost
        bottom -- The bottom proportion to fully boost

    Return:
        wrapper -- A function that return a full_boost curve of any size
    """
    def wrapper(n):
        """The wrapped function for the full_boost function

        Arguments:
            n -- The size of the vector to generate the full_boost to

        Returns:
            y -- An array of size n containing the full boost curve with the argument provided in the parent function
        """
        n1 = math.ceil(n * top)
        n2 = math.ceil(n * (1 - bottom))
        if n1 > n2:
            raise Exception("Overlapping veto")
        y = np.ones((n,))
        y[:n1] = -np.inf
        y[n2:] = np.inf
        return y
    return wrapper


if __name__ == '__main__':
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(*sigmoid_reciprocal()(500))
    plt.tight_layout()
    plt.savefig("img/s_curves.png")
