from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from typing import List, Tuple, Optional, Union
from uncertainty_testbed.uncertainty_models.uncertainty_base import ClassifierPosteriorBase
from uncertainty_testbed.utilities.functions import safe_logodds, sigmoid

__author__ = "nikos.daniilidis"


class Linear(keras.layers.Layer):
    """linear layer performing y = w.x + b without activation"""
    def __init__(self, units=1, kernel_initializer="glorot_uniform", bias_initializer="zeros"):
        super(Linear, self).__init__()
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer=self.bias_initializer, trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class HLALinear(keras.layers.Layer):
    """Linear classification layer with additive heteroscedastic noise."""
    def __init__(self, samples=100, units=1, kernel_initializer="glorot_uniform", bias_initializer="zeros",
                 activation=tf.keras.activations.sigmoid, sigma_scaling=0.1, **kwargs):
        super(HLALinear, self).__init__(**kwargs)
        self.samples = samples
        self.units = units
        self.linear = Linear(units=2*units, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.activation = activation
        self.sigma_scaling = sigma_scaling

    def call(self, inputs: Union[tf.Tensor, np.ndarray], training: Union[None, bool] = None):
        shape = inputs.shape.as_list()[:0] + [self.units, self.samples]  # [batch_size, units, samples]
        eps = tf.random.normal(shape=shape)
        x = self.linear(inputs)
        mu = x[..., :self.units]  # first half of linear is mus
        sigma = 1e-3 + tf.math.softplus(self.sigma_scaling * x[..., self.units:])  # second half with softplus is sigmas

        if training:
            logodds = tf.expand_dims(mu, 2) + tf.expand_dims(sigma, 2) * eps
            p = tf.reduce_mean(self.activation(logodds), axis=2)
            # self.add_loss(p) breaks the training step due to loss variable with size ().
            # We are passing p to the output during training instead.
            # self.add_loss(p)  # we are adding p instead of log(p), so we can use it in a custom loss during training
            return p, sigma  # note that we return class probability for mu, logit for sigma
        return self.activation(mu), sigma  # note that we return class probability for mu, logit for sigma


class HLAModel(keras.Model):
    """Custom Model class overriding the train step. Here we want to use the model.losses added by the HLALinear layer
    and combine it with the labels using the loss defined during compile (binary or categorical cross entropy).
    """
    def train_step(self, data):
        """"""
        x, y = data  # unpack the data

        with tf.GradientTape() as tape:
            y_pred, sigma = self(x, training=True)  # do the forward pass -- we will ignore y_pred in the loss
            loss = self.compiled_loss(y, y_pred)  # y_pred -> self.losses from HLALinear layer breaks the training step

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics -- here we will use y_pred # TODO this might break - we are not using the actual HLA loss
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metrics to their values
        return {m.name: m.result() for m in self.metrics}


class HLADropout(ClassifierPosteriorBase, ABC):
    """Keras classification model with MC Dropout sampling capability."""
    def __init__(self,
                 layer_tuple: Tuple[keras.layers.Layer, ...],
                 optimizer: keras.optimizers.Optimizer,
                 loss: str,
                 metrics: Tuple[str, ...],
                 name: str = "HLADropout"):
        """
                Initialize.
                :param layer_tuple (Tuple[keras.Layer, ...]): Tuple of Keras layers used to construct the model.
                :param optimizer (keras.optimizers.Optimizer):
                :param loss (str):
                :param metrics (Tuple[str, ...]):
                :param name (str):
                """
        # TODO: Add check that the layer tuple contains MC Dropout layers whenever having Dropout.
        assert isinstance(layer_tuple[-1], HLALinear)
        super().__init__()
        self.name = name
        self.layer_tuple = layer_tuple
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = [m for m in metrics]
        self.build_model()

    def build_model(self):
        inputs = self.layer_tuple[0]
        for ix, lar in enumerate(self.layer_tuple[1:-1]):
            if ix == 0:
                x = lar(inputs)
            else:
                x = lar(x)
        outputs = self.layer_tuple[-1](x)
        self.model = HLAModel(inputs=inputs, outputs=outputs, name=self.name)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            run_eagerly=False
        )

    def fit(self, x: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor], **kwargs):
        return self.model.fit(x, y, **kwargs)

    def predict(self, x: Union[np.ndarray, tf.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method to make the behavior of HLADropout models the same as Sequential models when calling
        model.predict(x)"""
        p, _ = self.model.predict(x)
        return np.concatenate([p, 1-p], axis=1)

    def summary(self, **kwargs):
        """Convenience method to make the behavior of HLADropout models the same as Sequential models when calling
        model.summary(...)"""
        self.model.summary(**kwargs)

    def sample_posterior_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the posterior probability and aleatoric uncertainty at input x."""
        if len(x.shape) == 1:  # if sending in a single row of x
            rows = 1
        else:
            rows = x.shape[0]
        xx = np.tile(x, (n, 1))
        yy, sg = self.model.predict(xx)
        if len(yy.shape) == 1:  # if sending in a single row of x and drawing only one sample
            outputs = yy.shape
        else:
            outputs = yy.shape[1]
        preds = np.reshape(yy, (n, rows, outputs), order='c')
        sigmas = np.reshape(sg, (n, rows, outputs), order='c')
        return preds, sigmas

    def posterior_mean_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the mean posterior probability and aleatoric uncertainty  at input x."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        return preds.mean(axis=0), sigmas.mean(axis=0)

    def posterior_mean_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the mean posterior logodds and aleatoric uncertainty  at input x."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        return preds_logodds.mean(axis=0), sigmas.mean(axis=0)

    def posterior_stddev_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the posterior probability and aleatoric uncertainty standard deviation at input x."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        return preds.std(axis=0), sigmas.std(axis=0)

    def posterior_stddev_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the posterior logodds and aleatoric uncertainty standard deviation at input x."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        return preds_logodds.std(axis=0), sigmas.std(axis=0)

    def posterior_percentile_proba(self, x: Union[np.ndarray, tf.Tensor], n: int, q: Tuple[float], **kwargs) \
            -> Union[np.array, float]:
        """Estimate the posterior probability percentiles q at input x. Assumes binary classification with sigmoid."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)  # (n, rows, outputs)
        noise = np.multiply(sigmas, np.random.normal(loc=0., scale=1., size=preds.shape))
        return np.percentile(preds, q, **kwargs), np.percentile(sigmoid(preds_logodds + noise), q, ** kwargs)

    def posterior_percentile_logodds(self, x: Union[np.ndarray, tf.Tensor], n: int, q: Tuple[float], **kwargs) \
            -> Union[np.array, float]:
        """Estimate the posterior logodds percentiles q at input x. Assumes binary classification with sigmoid."""
        preds, sigmas = self.sample_posterior_proba(x, n)
        preds_logodds = safe_logodds(preds)
        noise = np.multiply(sigmas, np.random.normal(loc=0., scale=1., size=preds.shape))
        return np.percentile(preds_logodds, q, **kwargs), np.percentile(preds_logodds + noise, q, ** kwargs)