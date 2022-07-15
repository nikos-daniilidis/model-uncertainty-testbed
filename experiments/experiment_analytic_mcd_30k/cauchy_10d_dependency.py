#!/usr/bin/env python
# coding: utf-8
"""Experiment looking at uncertainty estimation for a DNN model with Cauchy event generator, and producing
plots/metrics."""

__author__ = "nikos.daniilidis"

# In[3]:

from functools import partial
from uncertainty_testbed.generators.data_generator_explicit import AnalyticBinaryClassGenerator
from uncertainty_testbed.utilities.functions import map_to_constant
from tensorflow import keras
from experiments.experiment_utils.plot_utils import run_uncertainty_numbers
from experiments.model_templates.sequential_dropout_dnn_classifier import build_mcdropout_model

# In[2]:
dimensions = 10
event_name = "cauchy"
threshold = 0.5
noise_name = "cauchy"
noise_level = 0.01
n_all, n_train, n_val = 120096, 33280, 100096
s = partial(map_to_constant, c=noise_level)
eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=dimensions, name=event_name, threshold=threshold,
                                  noise_distribution=noise_name, noise_scale=s)
mcd_classifier = build_mcdropout_model(optimizer=keras.optimizers.Adamax(learning_rate=0.0084))

_ = run_uncertainty_numbers(eg, mcd_classifier, n_all, n_train, n_val, batch_size=8, epochs=6)

# In[10]:
dimensions = 10
event_name = "cauchy"
threshold = 0.5
noise_name = "cauchy"
num_train_g = 100096
num_train = 33280
num_val = 20000
noise_level = 0.1
s = partial(map_to_constant, c=noise_level)
eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=dimensions, name=event_name, threshold=threshold,
                                  noise_distribution=noise_name, noise_scale=s)
mcd_classifier = build_mcdropout_model(optimizer=keras.optimizers.Adamax(learning_rate=0.0056))

_ = run_uncertainty_numbers(eg, mcd_classifier, n_all, n_train, n_val, batch_size=8, epochs=6)

# In[11]:
#def generate_chisq_chisq_data(num_train=100000, num_val=20000, noise_level=0.):
dimensions = 10
event_name = "cauchy"
threshold = 0.5
noise_name = "cauchy"
n_all = 100096
n_train = 33280
n_val = 20000
noise_level = 0.3
# generate some data
s = partial(map_to_constant, c=noise_level)
eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=dimensions, name=event_name, threshold=threshold,
                                  noise_distribution=noise_name, noise_scale=s)
mcd_classifier = build_mcdropout_model(optimizer=keras.optimizers.Adamax(learning_rate=0.0056))

_ = run_uncertainty_numbers(eg, mcd_classifier, n_all, n_train, n_val, batch_size=8, epochs=6)
