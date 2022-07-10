"""Utilities used to generate plots in the experiments."""
__author__ = "nikos.daniilidis"


from tensorflow import keras
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from uncertainty_testbed.generators.data_generator_base import BinaryClassGeneratorBase


def plot_training_curves(history: keras.callbacks.History) -> None:
    """Plot loss and auc train metrics from history object"""
    # inspect metrics
    history_dict = history.history
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    auc = history_dict["auc"]
    val_auc = history_dict["val_auc"]
    epochs = range(1, len(loss)+1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "r--", label="Validation loss")
    plt.xlabel("Train Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(epochs, auc, "bo", label="Training auc")
    plt.plot(epochs, val_auc, "r--", label="Validation auc")
    plt.xlabel("Train Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()


def get_smoothed_scores_probs_preds(model: keras.Model, eg: BinaryClassGeneratorBase,
                                    x: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the latent scores and true probabilities from event generator eg, and predicted probabilities from trained
    model model, for input instances x. Apply smoothing with window n.
    :param model:
    :param eg:
    :param x:
    :param n:
    :return: true scores, true probabilities, predicted probabilities
    """
    p = model.predict(x)[:, 0]

    scores = eg.get_scores(x)
    ix_srt = np.argsort(scores)

    probs = eg.get_probabilities(x)[1]

    if n > 1:
        p = np.convolve(p[ix_srt[::-1]], np.ones(n)/n, mode='valid')
        scores = np.convolve(scores[ix_srt[::-1]], np.ones(n)/n, mode='valid')
        probs = np.convolve(probs[ix_srt[::-1]], np.ones(n)/n, mode='valid')

    return scores, probs, p


def plot_xy(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, style: str) -> None:
    """Single-line xy scatter plot utility."""
    plt.plot(x, y, style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_hist(x: np.ndarray, bins: int, xlabel: str, **kwargs) -> None:
    """Single-line histogram plot utility."""
    plt.hist(x, bins, **kwargs)
    plt.xlabel(xlabel)
    plt.show()


def plot_x1y1x2y2(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                  xlabel: str, ylabel: str, style1: str, style2: str) -> None:
    """Single line utility for scatter plot with two sets of x and y variables"""
    plt.plot(x1, y1, style1)
    plt.plot(x2, y2, style2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_hexbin(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, **kwargs) -> None:
    """Single line hexbinplot utility"""
    plt.hexbin(x, y, **kwargs)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_decision_data(data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], slice_i: int, slice_j: int,
                       model: keras.Model, eg: BinaryClassGeneratorBase,
                       figsize: Tuple[int, int], alpha_scatter: float) -> None:
    """Plot decision boundary and sample of data points"""
    x_train, y_train, x_val, y_val = data
    I, J = slice_i, slice_j
    x_min, x_max = x_val[:, I].min(), x_val[:, I].max()
    y_min, y_max = x_val[:, J].min(), x_val[:, J].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    x_ = 0. * np.ones((xxyy.shape[0], 10))
    x_[:, I] = xxyy[:, 0]
    x_[:, J] = xxyy[:, 1]
    z = model.predict(x_)
    z = z.reshape(xx.shape)
    y_ = eg.get_labels(x_)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.contourf(xx, yy, z, alpha=0.4)
    ixs = np.random.randint(0, y_.shape[0] - 1, 1000)
    ax.scatter(x_[ixs, I], x_[ixs, J], c=y_[ixs], s=20, alpha=alpha_scatter, edgecolor="k")
    plt.show()
