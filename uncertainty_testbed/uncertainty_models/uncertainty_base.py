import numpy as np
import tensorflow as tf
from typing import List, Optional, Union

__author__ = "nikos.daniilidis"


class ClassifierPosteriorBase(object):
    """Base class for classification models we use to predict uncertainty."""
    def __init__(self):
        pass

    def build_model(self):
        """Model definition."""
        raise NotImplementedError

    def fit(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor], **kwargs):
        """Fit the model on data."""
        raise NotImplementedError

    def sample_posterior_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> List[Optional[Union[np.ndarray, tf.Tensor]]]:
        """Sample from the posterior probability at input x."""
        raise NotImplementedError

    def posterior_mean_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the mean posterior probability at input x."""
        raise NotImplementedError

    def posterior_mean_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the mean posterior logodds at input x."""
        raise NotImplementedError

    def posterior_stddev_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the posterior probability standard deviation at input x."""
        raise NotImplementedError

    def posterior_stddev_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Union[np.ndarray, tf.Tensor]:
        """Estimate the posterior logodds standard deviation at input x."""
        raise NotImplementedError
