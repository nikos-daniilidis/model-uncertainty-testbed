from functools import partial
import numpy as np
from typing import Tuple, Union
from scipy.stats import uniform, norm, cauchy, chi2

__author__ = "nikos.daniilidis"


class BinaryClassGeneratorBase(object):
    """Base class for binary classification data set generators. Implements all reusable methods. """
    def __init__(self,
                 seed: int,
                 num_inputs: int,
                 name: str,
                 threshold: float,
                 score_fn: Union[callable, None],
                 noise_distribution: str,
                 noise_scale: callable = lambda x: np.ones(x.shape[0])):
        """
        Initialize the instance.
        :param seed: The seed for the numpy random state.
        :param num_inputs: Number of inputs in the input vector of each event (x dimension).
        :param name: Name of the type of event stream to generate.
        :param threshold: Float between 0. and 1.0. Fraction of the 0 class in the data.
        :param score_fn: Scoring function which computes the hidden scores for input instances x. Maps from
                        np.ndarray[:, :num_inputs] to np.ndarray[:]
        :param noise_distribution: String. Distribution of the noise used to scramble labels.
        :param noise_scale: Callable to determine the noise used to scramble labels. Defaults to scale(x) = 1.
        """
        assert noise_distribution in ('gauss', 'uniform', 'cauchy', 'chisq')
        self.seed = seed
        np.random.seed(seed)
        self.num_inputs = num_inputs
        self.name = name
        self.threshold = threshold
        self.score_fn = score_fn
        self.noise_distribution = noise_distribution
        self.noise_scale = noise_scale
        self.noise_params = {"uniform": (-0.5, 0.5),
                             "gauss": (0., 1.),
                             "cauchy": None,
                             "chisq": (num_inputs,)}

    def __get_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Get the hidden scores for a number of instances.
        :param x: Numpy array of float with shape (num_events, num_inputs).
        :return: Array with hidden scores for the rows in x. Shape is (num_events, ).
        """
        if self.score_fn is None:
            raise RuntimeError("Call to __get_scores() without prior call to set_score_fn().")
        return self.score_fn(x)

    def __get_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Noise function which computes additive, heteroscedastic noise for input instances x. The labeling noise at
        instance x will be noise_scale(x)*n, where n follows a uniform, normal, cauchy, or chisq distribution (this is
        determined by the noise_distribution attribute). The chisq distribution defaults to num_inputs degrees of
        freedom. The mean for all distributions is set to 0. The global scale of the noise is otherwise controlled by
        the noise_scale function from the constructor.
        :param x: Numpy array of float with shape (num_events, num_inputs).
        :return: Array with noise values for the rows in x. Shape is (num_events, ).
        """
        num_events = x.shape[0]
        if self.noise_distribution == "uniform":
            l, h = self.noise_params["uniform"]
            return np.multiply(np.random.default_rng().uniform(l, h, num_events), self.noise_scale(x))
        elif self.noise_distribution == "gauss":
            loc, scale = self.noise_params["gauss"]
            return np.multiply(np.random.default_rng().normal(loc, scale, num_events), self.noise_scale(x))
        elif self.noise_distribution == "cauchy":
            return np.multiply(np.random.default_rng().standard_cauchy(num_events), self.noise_scale(x))
        elif self.noise_distribution == "chisq":
            df = self.noise_params["chisq"][0]
            return np.multiply(np.random.default_rng().chisquare(df, num_events) - df, self.noise_scale(x))
        else:
            raise NotImplementedError("__get_noise() method not implemented for noise_distribution '{}'".
                                      format(self.noise_distribution))

    def get_labels(self, x: np.ndarray) -> np.ndarray:
        """
        Generate labels of class 0, 1 for input instances x. The labels are determined as:
        Y =  __score_fn(x) + noise_scale(x) >= threshold
        :param x: Numpy array of float with shape (num_events, num_inputs).
        :return: Array with class labels for the rows in x. Shape is (num_events, ).
        """
        y = np.greater_equal(self.__get_scores(x) + self.__get_noise(x), self.threshold)
        return y.astype(int)

    def generate_unlabeled(self, num_events: int) -> np.ndarray:
        """
        Generate unlabeled events (i.e. x's, input instances).
        :param num_events: How many data instances to generate.
        :return: Array of the instances, shape (num_events, num_inputs).
        """
        raise NotImplementedError

    def generate_labeled(self, num_events: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an instance of labeled events (inputs and labels).
        :param num_events: How many data instances to generate.
        :return: Tuple of numpy arrays with the inputs and outputs. First item shape is (num_events, num_inputs), second
                item shape (num_events, ).
        """
        x = self.generate_unlabeled(num_events)
        y = self.get_labels(x)
        return x, y

    def get_probabilities(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the probabilities of class 0, 1 for input instances x. The probabilities follow from:
        x |-> t = score(x) - threshold; x|-> s = scale(x)
        Y = (t + s*n >= 0) =>
            P(Y=1) = P(t + s*n >= 0) = P(n >= -t/s) = cdf(-t/s),
        where cdf is the cdf of the noise random variable n.
        :param x: The input instances, shape (num_events, num_inputs).
        :return: Array with class probabilities for the rows in x. Shape is (num_events, 2).
        """
        if self.noise_distribution == "uniform":
            l, h = self.noise_params["uniform"]
            cdf = partial(uniform.cdf, loc=(l+h)/2, scale=(h-l))
        elif self.noise_distribution == "gauss":
            loc, scale = self.noise_params["gauss"]
            cdf = partial(norm.cdf, loc=loc, scale=scale)
        elif self.noise_distribution == "cauchy":
            cdf = partial(cauchy.cdf, loc=0, scale=1)
        elif self.noise_distribution == "chisq":
            df = self.noise_params["chisq"][0]
            cdf = partial(chi2.cdf, df=df)
        else:
            raise NotImplementedError("__get_noise() method not implemented for noise_distribution '{}'".
                                      format(self.noise_distribution))
        t = self.__get_scores(x) - self.threshold
        p0 = cdf(-np.divide(t, self.noise_scale(x)))
        return p0, 1-p0

    def local_entropy(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the local entropy of the label distribution given the latent score, at each input instance x.
        The entropy is H(Y|h) = P0 * log(P0) + P1 * log(P1), where P0 = P[Y=0 | h, s, x], where h is the hidden score,
        returned by the __score_fn, s is the scale of the noise random variable returned by noise_scale.
        This entropy is the asymptotic log-loss, for input instance x, of a model from a hypothesis space which contains
        the __score_fn.
        :param x: The input instances, shape (num_events, num_inputs).
        :return: Array with entropy values for the rows in x. Shape is (num_events,).
        """
        p0, p1 = self.get_probabilities(x)
        return np.multiply(p0, np.log(p0)) + np.multiply(p1, np.log(p1))
