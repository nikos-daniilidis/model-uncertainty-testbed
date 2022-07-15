"""Utilities used to generate plots in the experiments."""

__author__ = "nikos.daniilidis"


from tensorflow import keras
from typing import Tuple, Iterable
from matplotlib import pyplot as plt
import numpy as np
from experiments.experiment_utils.train_utils import generate_full
from uncertainty_testbed.generators.data_generator_base import BinaryClassGeneratorBase
from uncertainty_testbed.uncertainty_models.uncertainty_base import ClassifierPosteriorBase


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
    :return: true scores, true probabilities, posterior lower percentile, posterior mean, posterior higher percentile
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


def get_smoothed_scores_probs_post_preds(model: ClassifierPosteriorBase, eg: BinaryClassGeneratorBase,
                                         x: np.ndarray, ptiles: Tuple[float, float], n: int, s: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the latent scores, true probabilities from event generator eg, and sampled posterior percentiles, probabilities
    from trained model model, for input instances x. Apply smoothing with window n.
    :param model: Trained model.
    :param eg: Event generator object.
    :param x: Input instances.
    :param ptiles: Percentiles to retunr for the posterior probability samples.
    :param n: Window to use for smoothing the return values.
    :param s: Number of posterior samples to use in estimates.
    :return: true scores, true probabilities, predicted probabilities.
    """
    assert len(ptiles) == 2
    post_lo_mean = model.posterior_mean_logodds(x, n=s)

    p = 1 / (1 + np.exp(-post_lo_mean[:, 0]))
    ptiles = model.posterior_percentile_proba(x, n=s, q=ptiles, axis=0)
    p_l = ptiles[0, :, 0]
    p_h = ptiles[1, :, 0]

    scores = eg.get_scores(x)
    ix_srt = np.argsort(scores)

    probs = eg.get_probabilities(x)[1]

    if n > 1:
        p = np.convolve(p[ix_srt[::-1]], np.ones(n)/n, mode='valid')
        p_l = np.convolve(p_l[ix_srt[::-1]], np.ones(n) / n, mode='valid')
        p_h = np.convolve(p_h[ix_srt[::-1]], np.ones(n) / n, mode='valid')
        scores = np.convolve(scores[ix_srt[::-1]], np.ones(n)/n, mode='valid')
        probs = np.convolve(probs[ix_srt[::-1]], np.ones(n)/n, mode='valid')
    else:
        p = p[ix_srt[::-1]]
        p_l = p_l[ix_srt[::-1]]
        p_h = p_h[ix_srt[::-1]]
        scores = scores[ix_srt[::-1]]
        probs = probs[ix_srt[::-1]]

    return scores, probs, p_l, p, p_h


def plot_xys(x: np.ndarray, ys: Iterable[Tuple[np.ndarray, str]],
             xlabel: str, ylabel: str) -> None:
    """
    Single-line xy scatter plot utility. Plot multiple ys using common x.
    :param x: shared x values
    :param ys: iterable of tuples, first element in each tuple is an array with y values, second element is
            the style string to use in the plot
    :param xlabel:
    :param ylabel:
    :param style:
    :return:
    """
    for y, stl in ys:
        plt.plot(x, y, stl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_xys_and_fillbetween(x: np.ndarray, ys: Iterable[Tuple[np.ndarray, str]],
                             y_l: np.ndarray, y_h: np.ndarray, xlabel: str, ylabel: str, **kwargs):
    """Plot multiple ys using common x, and overlay a fillbetween plot.
    :param x: shared x values
    :param ys: iterable of tuples, first element in each tuple is an array with y values, second element is
            the style string to use in the plot
    :param y_l: low values for fillbwetween
    :param y_h: high values for fillbetween
    :param xlabel:
    :param ylabel:
    :param kwargs: kwargs for fillbetween
    :return:
    """
    for y, stl in ys:
        plt.plot(x, y, stl)
    plt.fill_between(x, y_l, y_h, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_hist(x: np.ndarray, xlabel: str, **kwargs) -> None:
    """Single-line histogram plot utility."""
    plt.hist(x, **kwargs)
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


def run_uncertainty_numbers(eg: BinaryClassGeneratorBase, model: keras.Model, n_all: int, n_train: int, n_val: int,
                            **kwargs) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param eg:
    :param model:
    :param n_all:
    :param n_train:
    :param n_val:
    :param kwargs: kwargs for model.fit(). Must specify batch size and epochs.
    :return:
    """
    # generate data sets
    data = generate_full(eg, n_all, n_train, n_val)
    x_train, y_train, x_val, y_val = data
    noise_level = eg.noise_scale(x_train[0,:])

    # In[3]:
    model.fit(
        x_train,
        y_train,
        **kwargs,
        validation_data=(x_val, y_val)
    )

    # In[4]:
    scores, probs, p_l, p, p_h = get_smoothed_scores_probs_post_preds(model, eg,
                                                                      x_val[:100, :], (5., 95.), 1, 300)

    # In[5]:
    from typing import Tuple, Iterable

    plot_xys_and_fillbetween(scores, ((p, '--b'),), p_l, p_h, 'latent scores', 'p_hat',
                             alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    # In[6]:
    plot_xys_and_fillbetween(probs, ((probs, '--g'), (p, '--b')), p_l, p_h, 'p', 'p_hat',
                             alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    # In[7]:
    # find percentage of 95% confidence intervals containing the true probability
    s = 300  # number of posterior samples to sample for posterior for each x
    d = 10000  # number of x samples points to use
    scores, probs, p_l, p, p_h = get_smoothed_scores_probs_post_preds(model, eg,
                                                                      x_val[:d, :], (5., 95.), 1, s)
    #logodds = safe_logodds(probs)

    # In[8]:
    inside = np.logical_and(np.less_equal(p_l, probs), np.less_equal(probs, p_h))
    print("For noise level: {}: {}% within 90% CI".format(noise_level, 100 * np.mean(inside)))
    normed_p_dist = (p - probs) / (1e-16 + p_h - p_l)
    plot_hist(normed_p_dist, bins=np.linspace(-3, 3), xlabel='(p_hat - p) / (p95-p5)', color='g', alpha=0.5)

    # In[9]:

    thr = 0.2
    print("Percentage of 90% CI below {}: {}".format(thr, np.mean(np.less_equal(p_h - p_l, thr))))
    print("Average p95: {}".format(np.mean(p_h)))

    plot_hist(p_h - p_l, bins=np.linspace(0., 1.), xlabel='p95-p5', color='g', alpha=0.5)

    return data
