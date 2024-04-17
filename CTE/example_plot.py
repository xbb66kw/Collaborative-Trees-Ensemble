#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:33:40 2023

@author: xbb
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
run = False
if run and __name__ == '__main__':
    
#%%
    n = 500
    p = 10
    rho = 0.1
    sample_var = 1

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
        X[i, :] = norm.cdf(np.dot(Q[i, :], A))  # half of the mass is before zero



    y = 5 * X[:, 0] + 20 * (X[:, 2] - 0.5)**2 + 15 * X[:, 4] + \
        + 2 * X[:, 8] \
        + 10 * np.sin(np.pi * (X[:, 8] - 0.5) * (X[:, 9] - 0.5))


    y = y + np.squeeze(np.random.multivariate_normal(\
                                [0], [[sample_var]], size = y.shape[0]))
    #%%
    # If we need to find out the best parameter set
    
    
    # 1. `transformed = True` without `group_list` then all features 
    #   are solo features, and are transformed into groups.
    # 2. `n_bins` is one of the hyperparameters.    
    # 3. `max_levels` is the number of rounds for hyperparameter 
    # searching.
    best_param = search_start(
        X, y, max_evals = 20, transformed = True)#, n_trees_user = 10)

    #%%
    ####
    # Several saved configurations for our Examples:
    
    # y = 5 * X[:, 0] + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 4] + \
    #    + 2 * X[:, 8] \
    #    + 10 * (X[:, 8] -0.5) * (X[:, 9] - 0.5)
    # rho = 0.1
    best_param_1 = {'n_trees': 12,
     'min_samples_split': 10,
     'min_samples_leaf': 0,
     'random_update': 0.0,
     'alpha': float('Inf'),
     'max_depth': 10,
     'n_bins': None}
    
    # with binning
    best_param_1_g = {'n_trees': 12,
     'min_samples_split': 10,
     'min_samples_leaf': 0,
     'random_update': 0.0,
     'alpha': float('Inf'),
     'max_depth': 10,
     'n_bins': 5}
    
    # a single tree
    best_param_1_single = {'n_trees': 1,
     'min_samples_split': 10,
     'min_samples_leaf': 0,
     'random_update': 0.0,
     'alpha': float('Inf'),
     'max_depth': 10,
     'n_bins': 5}
    
    # rho = 0.8
    # the same as `with binning`
    best_param_1_bias = {'n_trees': 12,
     'min_samples_split': 10,
     'min_samples_leaf': 0,
     'random_update': 0.0,
     'alpha': float('Inf'),
     'max_depth': 10,
     'n_bins': 5}
    
    

    #%%
    # Use the one parameter set of interest
    print('Remember to change the parameter configuration.')
    used_param = best_param_1_g
    forest = CollaborativeTreesEnsemble(n_estimators = 100,
        dict_param = used_param) #
    
    print('Best indices for tuning parameters for CTE: ',
          forest.get_info())
    
    #%%

    # Fit the model
    forest.multi_fit(X, y)
    #%%
    #####
    # Call plot_network_start() to plot the network diagram
    # Table object for plotting
    obj_table = forest.diagram_pack
    
    # obj_table = obj[1]
    # keep the information for only the first 10 features.
    obj_table_temp = obj_table
    obj_table_temp[0][:10]
    obj_table_temp[1][:10]
    obj_table_temp[2][:, :10][:10,:]
    obj_table_temp[3][:, :10][:10,:]
    obj_table_temp[4]
    obj_table = []
    obj_table.append(obj_table_temp[0][:10])
    obj_table.append(obj_table_temp[1][:10])
    obj_table.append(obj_table_temp[2][:, :10][:10,:])
    obj_table.append(obj_table_temp[3][:, :10][:10,:])
    
    
    # rf.feature_importances_
    # obj_table = []
    # obj_table.append(rf.feature_importances_)
    
    # p_ = len(rf.feature_importances_)
    # obj_table.append(0.5*np.ones(p_))
    # obj_table.append(np.zeros(p_**2).reshape(p_, p_))
    # obj_table.append(np.zeros(p_**2).reshape(p_, p_))
    
    #%%
    ######            
    # obj_table : a list
        # obj_table[0] is the standardized XMDI_{i}'s
        # obj_table[1] is the standardized XMDI_{ii}'s
        # obj_table[2] is XMDI_{ij}'s (XMDI_{ii}'s are zero)
        # obj_table[3] is XMDI_{ij}'s / XMDI_{ii}'s    ######
    
    ####
    # Parameter for the network diagram
    parameters = {}
    # Change these bases to make figures fit better
    # base of node sizes
    parameters['base_size'] = 8500
    # base of edges' sizes
    parameters['base_edge_size'] = 10
    # Positiions of labels
    parameters['horizontal_positive_shift'] = 0.1
    parameters['horizontal_negative_shift'] = 0.1
    parameters['vertical_positive_shift'] = 0.
    parameters['vertical_negative_shift'] = 0.1
    parameters['label_font_size'] = 40
    parameters['edge_label_font_size'] = 25
    
    #####
    # Draw the network diagram
    # Save the png file from the python figure window
    plot_network_start(obj_table, parameters, digits = 1, 
                       colorbar= True)
    
    