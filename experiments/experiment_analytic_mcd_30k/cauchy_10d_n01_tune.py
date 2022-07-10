#!/usr/bin/env python
# coding: utf-8
"""Experiment performing hyperparameter scan for a DNN model with Cauchy event generator, and producing plots/metrics
for the winning model."""

__author__ = "nikos.daniilidis"

# In[1]:
from functools import partial
from uncertainty_testbed.generators.data_generator_explicit import AnalyticBinaryClassGenerator
from uncertainty_testbed.utilities.functions import map_to_constant
from experiments.experiment_utils.train_utils import generate_full, hyperparameter_scan, train_from_params
from experiments.experiment_utils.plot_utils import plot_training_curves, get_smoothed_scores_probs_preds, plot_xy, \
    plot_hist, plot_x1y1x2y2, plot_hexbin, plot_decision_data
from experiments.model_templates.sequential_dropout_dnn_classifier import SequentialDropout_10_60_025_x3_Template

dimensions = 10
event_name = "cauchy"
threshold = 0.5
noise_name = "cauchy"
noise_level = 0.1
n_all, n_train, n_val = 120096, 33280, 100096

# generate data sets
s = partial(map_to_constant, c=noise_level)
eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=dimensions, name=event_name, threshold=threshold,
                                  noise_distribution=noise_name, noise_scale=s)
data = generate_full(eg, n_all, n_train, n_val)
x_train, y_train, x_val, y_val = data

# In[3]:
template = SequentialDropout_10_60_025_x3_Template()  # define model structure and hyperparameter scan schedule
champion_params, scan_results = hyperparameter_scan(template, data)  # perform hyperparameter scan, return winning parameters

# In[4]:
# In[5]:
model, history = train_from_params(champion_params, template, data)

# In[6]:
plot_training_curves(history)  # plot loss and auc train metrics from history object.

# In[7]:
scores, probs, p = get_smoothed_scores_probs_preds(model, eg, x_val[:1000, :], 5)

# In[8]:
plot_xy(scores, p, 'latent scores', 'p_hat', '--b')
plot_hist(scores, 50, 'latent scores')

# In[9]:
plot_x1y1x2y2(probs, probs, probs, p, 'p', 'p_hat', '--g', '--b')
plot_hist(x=probs, bins=50, xlabel='p', color='g', alpha=0.5)

# In[10]:
scores, probs, p = get_smoothed_scores_probs_preds(model, eg, x_val[:10000, :], 1)
plot_hexbin(x=probs, y=p, xlabel='p', ylabel='p_hat', gridsize=30, bins='log')

# In[11]:
plot_decision_data(data, slice_i=2, slice_j=6, model=model, eg=eg, figsize=(6,6), alpha_scatter=0.3)

# In[12]:
model.summary()

# In[13]:
print (champion_params)

# In[14]:
print (scan_results)

