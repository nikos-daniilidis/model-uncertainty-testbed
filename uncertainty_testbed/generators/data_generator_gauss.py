from abc import ABC
import numpy as np
from scipy.stats import chi2, cauchy, norm
from uncertainty_testbed.generators.data_generator_base import BinaryClassGeneratorBase

__author__ = "nikos.daniilidis"


class AnalyticBinaryClassGenerator(BinaryClassGeneratorBase, ABC):
    """
    Class which generates labeled data following analytic functional forms. The features follow iid gaussian
    distributions. The labels are determined as:
        gauss: \sum_{i=1}^{n}(c_i \dot x_i) + noise > t,
        chisq: \sum_{i=1}^{n}(x_i^2) + noise > t,
        cauchy: \sum_{i=1}^{n/2}(x_i)/\sum_{i=n/2+1}^{n}(x_i) + noise > t,
    """
    def __init__(self,
                 seed: int,
                 num_inputs: int,
                 name: str,
                 threshold: float,
                 noise_distribution: str,
                 noise_scale: callable = lambda x: np.ones(x.shape[0])):
        """
        Initialize the instance.
        :param seed: The seed for the numpy random state.
        :param num_inputs: Number of inputs in the input vector of each event (x dimension).
        :param name: Name of the type of event stream to generate.
        :param threshold: Float between 0. and 1.0. Fraction of the 0 class in the data.
        :param noise_distribution: String. Distribution of the noise used to scramble labels.
        :param noise_scale: Callable to determine the noise used to scramble labels. Defaults to scale(x) = 1.
        """
        assert isinstance(seed, int)
        assert isinstance(num_inputs, int)
        assert name in ('gauss', 'chisq', 'cauchy')
        assert isinstance(threshold, float)
        assert (threshold >= 0.) and (threshold <= 1.)
        super().__init__(seed=seed, num_inputs=num_inputs, name=name, threshold=threshold,
                         noise_distribution=noise_distribution, noise_scale=noise_scale, score_fn=None)
        self.threshold = self.__get_threshold(threshold)
        self.score_fn = self.__score_fn

    def __get_threshold(self, t: float) -> float:
        """
        Calculate the threshold value which will result in class balance equal to t.
        :param t: The class balance at a threshold threshold
        :return: threshold
        """
        if self.name == 'gauss':
            self.coefficients = 2. * np.random.random(size=self.num_inputs) - 1.  # random coefficients in range [-1, 1]
            sigma = np.sqrt(np.sum(np.power(self.coefficients, 2)))  # standard deviation of the latent score
            threshold = sigma * norm.isf(t)
        elif self.name == 'chisq':
            threshold = chi2.isf(t, df=self.num_inputs)
        elif self.name == 'cauchy':
            assert (self.num_inputs % 2 == 0)  # only implemented for even number of inputs
            self.n_top = int(self.num_inputs / 2)
            self.numerator_indices = range(0, self.n_top)
            self.denumerator_indices = range(self.n_top, self.num_inputs)
            self.cauchy = cauchy(0., np.sqrt(self.num_inputs) / np.pi)
            threshold = self.cauchy.isf(t)
        else:
            raise NotImplementedError("name argument must be 'gauss', 'chisq', or 'cauchy'.")
        return threshold

    def __score_fn(self, x: np.ndarray) -> np.ndarray:
        if self.name == "gauss":
            return self.__gauss_score(x)
        elif self.name == "cauchy":
            return self.__cauchy_score(x)
        elif self.name == "chisq":
            return self.__chisq_score(x)
        else:
            raise NotImplementedError("__score_fn is only implemented for gauss, cauchy, and chisq.")

    def generate_unlabeled(self, num_events: int) -> np.ndarray:
        """
        Generate unlabeled events (input instances).
        :param num_events: How many data instances to generate.
        :return: Array of the instances, shape (num_events, num_inputs).
        """
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        x = nx.reshape((num_events, self.num_inputs))
        return x

    def __gauss_score(self, x: np.ndarray) -> np.ndarray:
        """
        Latent score for a stream of gaussian type events. The score is sum of coefficients*inputs.
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :return: Numpy arrays of float.
        """
        return np.inner(x, self.coefficients)

    def __cauchy_score(self, x: np.ndarray) -> np.ndarray:
        """
        Latent score for a stream of cauchy type events. The score is (sum of inputs[:,top])/(sum of inputs[:,bottom]).
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :return: Numpy arrays of float.
        """
        return np.divide(np.sum(x[:, self.numerator_indices], axis=1),
                         np.sum(x[:, self.denumerator_indices], axis=1))

    def __chisq_score(self, x: np.ndarray) -> np.ndarray:
        """
        Latent score for a stream of chisq type events. The score is (sum of square(input).
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :return: Numpy arrays of float.
        """
        return np.sum(np.square(x), axis=1)


def mini_check(name, noise_distribution, noise_magnitude):
    s = lambda x: noise_magnitude * np.ones(x.shape[0])
    eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=10, name=name, threshold=0.5,
                                      noise_distribution=noise_distribution, noise_scale=s)
    x, y = eg.generate_labeled(10000)
    print('{} ({} noise) at 0.5 ({}) -> {}'.format(name, noise_distribution, noise_magnitude, np.mean(y)))


def basic_checks():
    for name in ['gauss', 'cauchy', 'chisq']:
        for noise_distribution in ['uniform', 'gauss', 'cauchy', 'chisq']:
            for scale in [0., 0.01, 0.1, 1.]:
                mini_check(name, noise_distribution, scale)


if __name__ == "__main__":
    basic_checks()