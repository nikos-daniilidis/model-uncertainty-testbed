"""Event generator using Variational Auto Encoder."""

__author__ = "nikos.daniilidis"


from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow import keras
from uncertainty_testbed.generators.event_generator_base import TrainableEventGeneratorBase

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    __author__ = "fchollet"

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEGenerator(TrainableEventGeneratorBase, ABC):
    """Variational auto encoder event generator"""
    def __init__(self, encoder_layers, decoder_layers, **kwargs):
        super(VAEGenerator, self).__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def encoder(self):
        encoder_inputs = self.encoder_layers[0]
        for i, l in enumerate(self.encoder_layers[1:-2]):  # by construction the first layer is Input, the last two are output
            if i == 0:
                x = l(encoder_inputs)
            else:
                x = l(x)
        z_mean = self.encoder_layers[-2](x)
        z_log_var = self.encoder_layers[-1](x)




    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
