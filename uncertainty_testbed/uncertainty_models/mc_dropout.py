"""Implementation of classifier posterior base using MC Dropout."""

__author__ = "nikos.daniilidis"

from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Optional, Union
from uncertainty_testbed.uncertainty_models.uncertainty_base import ClassifierPosteriorBase
from uncertainty_testbed.utilities.functions import safe_logodds


class MCDropoutLayer(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCDropoutKerasClassification(ClassifierPosteriorBase, ABC):
    """Keras classification model with MC Dropout sampling capability."""
    def __init__(self,
                 layer_tuple: Tuple[keras.layers.Layer, ...],
                 optimizer: keras.optimizers.Optimizer,
                 loss: str,
                 metrics: Tuple[str, ...],
                 name: str = "MCDropout"):
        """
        Initialize.
        :param layer_tuple (Tuple[keras.Layer, ...]): Tuple of Keras layers used to construct the model.
        :param optimizer (keras.optimizers.Optimizer):
        :param loss (str):
        :param metrics (Tuple[str, ...]):
        :param name (str):
        """
        # TODO: Add check that the layer tuple contains MC Dropout layers whenever having Dropout.
        super().__init__()
        self.name = name
        self.layer_tuple = layer_tuple
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = [m for m in metrics]
        self.model = keras.models.Sequential()
        self.build_model()

    def build_model(self):
        for l in self.layer_tuple:
            self.model.add(l)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics)

    def fit(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor], **kwargs):
        self.model.fit(x, y, **kwargs)

    def sample_posterior_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Sample from the posterior probability at input x."""
        if len(x.shape) == 1:  # if sending in a single row of x
            rows = 1
        else:
            rows = x.shape[0]
        xx = np.tile(x, (n, 1))
        yy = self.model.predict(xx)
        if len(yy.shape) == 1:  # if sending in a single row of x and drawing only one sample
            outputs = yy.shape
        else:
            outputs = yy.shape[1]
        preds = np.reshape(yy, (n, rows, outputs), order='c')
        return preds

    def posterior_mean_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the mean posterior probability at input x."""
        preds = self.sample_posterior_proba(x, n)
        return preds.mean(axis=0)

    def posterior_mean_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the mean posterior logodds at input x."""
        preds = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        return preds_logodds.mean(axis=0)

    def posterior_stddev_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Union[np.ndarray, tf.Tensor]:
        """Estimate the posterior probability standard deviation at input x."""
        preds = self.sample_posterior_proba(x, n)
        return preds.std(axis=0)

    def posterior_stddev_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Union[np.ndarray, tf.Tensor]:
        """Estimate the posterior logodds standard deviation at input x."""
        preds = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        return preds_logodds.std(axis=0)

    def posterior_percentile_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, q: Tuple[float, float], **kwargs) \
            -> Union[np.array, float]:
        """Estimate the posterior probability percentiles q at input x."""
        preds = self.sample_posterior_proba(x, n)
        return np.percentile(preds, q, **kwargs)

    def posterior_percentile_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, q: Tuple[float, float], **kwargs) \
            -> Union[np.array, float]:
        """Estimate the posterior logodds percentiles q at input x. Assumes binary classification with sigmoid."""
        # TODO: Generalize to handle multi-class and non-sigmoid activation.
        preds = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        return np.percentile(preds_logodds, q, **kwargs)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000, 28*28)
    X_train = X_train.astype("float32")/255
    X_test = X_test.reshape(10000, 28 * 28)
    X_test = X_test.astype("float32") / 255
    print(X_train.shape, y_train.shape)
    print(type(X_train), type(y_train))
    print(y_train.min(), y_train.max(), y_train.mean())
    print(type(X_train[0, 0]), type(y_train[0]))

    layers = (
        MCDropoutLayer(0.25),
        keras.layers.Dense(300, activation="relu"),
        MCDropoutLayer(0.25),
        keras.layers.Dense(300, activation="relu"),
        MCDropoutLayer(0.25),
        keras.layers.Dense(10, activation="softmax")
    )
    mcd_classifier = MCDropoutKerasClassification(
        layer_tuple=layers,
        optimizer=keras.optimizers.Nadam(lr=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=("accuracy",),
        name="MCDropout"
    )
    mcd_classifier.fit(X_train, y_train, epochs=4, batch_size=128, validation_data=(X_test, y_test))

    samples = mcd_classifier.sample_posterior_proba(X_test[1:2, :], n=10)
    print("Sampled probabilities: {}".format(' '.join(str(samples))))

    post_mean = mcd_classifier.posterior_mean_proba(X_test[1:3, :], n=100)
    post_std = mcd_classifier.posterior_stddev_proba(X_test[1:3, :], n=100)
    print("Posterior probability mean(sd): {}({})".format(post_mean, post_std))

    post_lo_mean = mcd_classifier.posterior_mean_logodds(X_test[1:3, :], n=100)
    post_lo_std = mcd_classifier.posterior_stddev_logodds(X_test[1:3, :], n=100)
    print("Posterior logodds mean(sd): {}({})".format(post_lo_mean, post_lo_std))

    post_lo_percemtile = mcd_classifier.posterior_percentile_logodds(X_test[1:3, :], n=300)

