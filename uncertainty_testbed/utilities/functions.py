"""General utility  functions to use throughout."""
import numpy as np


def map_to_constant(x: np.ndarray, c: float) -> np.ndarray:
    """
    Return an array mapping all rows in array x to the same constant, c
    :param x: Array of inputs
    :param c: Floating point constant
    :return: Array with all values c having as many rows as x
    """
    return c * np.ones(x.shape[0])


def safe_logodds(p: np.ndarray, lim: float = 1e-16) -> np.ndarray:
    """
    Compute the natural logarithm log-odds of p, with a cutoff lim for probabilities very close to 0 or 1
    :param p: Array of probabilities.
    :param lim: Floating point constant
    :return: Array of log odd values
    """
    return np.log((p + lim) / (1 - p + lim))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid  of x
    :param x: Array of log odds.
    :return: Array of probabilities.
    """
    return 1 / (1 + np.exp(-x))