#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:09:46 2024

This example demonstrates how to train a Collaborative Trees Ensemble
model, tune its hyperparameters, and visualize the learned 
relationships.

@author: xbb

mypy PATH/example.py --ignore-missing-imports
"""
#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from scipy.stats import norm

from method.cte import CollaborativeTreesEnsemble 
from method.util.param_search import search_start
from method.util.plot_network import plot_network_start
#%%
if __name__ == '__main__': 
#%%
    # Generating simulated data
    n = 500 # size of the simulated data
    p = 10 # number of the input explanatory features
    rho = 0. # AR(1) correlation level between features
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho**abs(i - j)
    X = np.empty(n * p).reshape(n, p)
    # Gaussian corpula distribution
    Q = np.random.multivariate_normal(\
        [0], [[1]], size = n * p).reshape(n, p)
    A = np.linalg.cholesky(cov)
    for i in range(n):
        X[i, :] = norm.cdf(np.dot(Q[i, :], A)) 
        
    y = 5 * X[:, 0] + 10 * X[:, 1] + 20 * (X[:, 2] - 0.5)**2 \
        + 10 * np.sin(np.pi * X[:, 3] * X[:, 4])
        
    # Incorporate model error with univariate variance into y
    y = y + np.squeeze(np.random.multivariate_normal(\
                            [0], [[1]], size = y.shape[0]))
    #%%
    # Find out the best parameter set.
    
    # 1. `transformed = True` without `group_list` then all features
    #   are solo features, and are transformed into groups.
    # 2. Here, [0, 1], [4, 5, 6], [8, 9] are existing groups and are 
    #   not further processed when transformed = True. Other features
    #   are solo features and are binned into groups.
    # 3. `n_bins` is one of the hyperparameters.    
    # 4. `max_levels` is the number of rounds for hyperparameter 
    #   searching.
    best_param = search_start(
        X, y, max_evals = 20, transformed = True)

    #%%
    # Two samples configuration samples.
    
    # Without grouping (`n_bins` is None)
    best_param = {'n_trees': 9,
     'min_samples_split': 15,
     'min_samples_leaf': 20,
     'random_update': 0.0001,
     'alpha': 10000.0,
     'max_depth': 10,
     'n_bins': None}
    # With feature grouping (`n_bins` is not None)
    best_param_g = {'n_trees': 7,
     'min_samples_split': 20,
     'min_samples_leaf': 5,
     'random_update': 0.0001,
     'alpha': 100,
     'max_depth': 30,
     'n_bins': 7}
    #%%
    # Use the the parameter set `used_param`
    used_param = best_param
    
    # Initialize the Collaborative Trees Ensemble model.
    # When `n_bins` is not None, solo features are transformed into
    # groups.
    forest = CollaborativeTreesEnsemble(n_estimators = 100,
        dict_param = used_param)
    #%%
    # Fit the Collaborative Trees Ensemble model.
    # Without specifying `group_list`. Hence, all features are
    # transformed into gorups when `n_bins` is not None
    forest.multi_fit(X, y)

    # Alternatively, fit the model without using multiprocess.
    # forest.fit(X, y)
    #%%

    # Generate network diagram
    obj_table = forest.diagram_pack
    obj_table[4] # Details of obj_table
    parameters = {'base_size': 8500, 'base_edge_size': 4,
                  'horizontal_positive_shift': 0.1,
                  'horizontal_negative_shift': 0.0,
                  'vertical_positive_shift': 0.15,
                  'vertical_negative_shift': -0.1,
                  'label_font_size': 40, 'edge_label_font_size': 25}
    plot_network_start(obj_table, parameters, digits = 1)
    