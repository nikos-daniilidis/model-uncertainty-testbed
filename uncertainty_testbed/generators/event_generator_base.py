"""Base class for trainable event generator. Inherits from keras.Model class and includes sample() method."""

__author__ = "nikos.daniilidis"

import numpy as np
from tensorflow import keras


class TrainableEventGeneratorBase(keras.Model):
    def sample(self, num_events: int) -> np.ndarray:
        """Method used to sample from the event generating distribution. Will be passed to the generate_unlabeled
        method of DataGenerator, so signatures need to match."""
        raise NotImplementedError