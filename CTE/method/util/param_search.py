#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:26:51 2023

@author: xbb
"""
import numpy as np
# from sklearn.model_selection import train_test_split
from hyperopt import Trials, fmin, tpe
# from util.to_factor_list import X_to_group_of_indicators
from method.cte import CollaborativeTreesEnsemble
from method.util.parameter_tuning_space import p_s, \
    space_cte, objective_cte_regression
    
def search_start(X: np.ndarray,
                 y: np.ndarray,
                 n_trees_user: int | None = None,
                 max_evals: int = 100,
                 group_list: list[list[int]] | None = None,
                 transformed: bool = False,
                 multiprocessing: int = 1) -> dict:
    '''
    

    Parameters
    ----------
    X : np.ndarray
        Sample feature matrix.
    y : np.ndarray
        Sample response vector.
    n_trees_user : int | None, optional
        Fix the number of collaborative trees. The default is None.
    max_evals : int, optional
        The number of optimization runs for hyperparameter tuning. 
        The default is 100.
    group_list : list[list[int]] | None, optional
        Specify the information of group of indicators for training 
        models. The default is None, which means all features are 
        solo features [[1], [2], [3], ...].
    transformed : bool, optional
        If the value is false, `n_bins` is set to None, which means
        no feature bining is needed during model training. The 
        default is False.
    multiprocessing : int, optional
        `multiprocessing` specifies the number of CPUs used for 
        multiprocessing. The default is 1
    Returns
    -------
    dict
        A dictionary with the selected hyperparameters. 
        n_bins = None means no feature bining is needed during model 
        training. 

    '''
    
    
    
    # To use `fmin` with fn = `objective_cte_regression`
    # you need to specify these parameters, with their defaults
    # given below.
    # Default value is None
    space_cte['n_trees_user'] = n_trees_user
    # Default value is None
    space_cte['group_list'] = group_list
    space_cte['transformed'] = transformed    
    space_cte['multiprocessing'] = multiprocessing
    
    space_cte['X'] = X
    space_cte['y'] = y

    trials = Trials()
    best_hyperparams_cte = fmin(fn = objective_cte_regression,
                            space = space_cte,
                            trials = trials,
                            algo = tpe.suggest,
                            max_evals = max_evals)
    
    if space_cte['n_trees_user'] is not None:
        m_collaborative_trees = space_cte['n_trees_user']
    else:
        m_collaborative_trees = \
            p_s['n_trees'][best_hyperparams_cte['n_trees']]
    
    forest = CollaborativeTreesEnsemble(n_estimators = 1,
        m_collaborative_trees = m_collaborative_trees,
        random_update = 
          p_s['random_update'][best_hyperparams_cte['random_update']], 
        min_samples_split =
          p_s['min_samples_split'][best_hyperparams_cte['min_samples_split']], 
        min_samples_leaf = 
          p_s['min_samples_leaf'][best_hyperparams_cte['min_samples_leaf']],
        alpha = 
          p_s['alpha'][best_hyperparams_cte['alpha']],
        max_depth =
          p_s['max_depth'][best_hyperparams_cte['max_depth']]) #

    output = forest.get_info()
    n_bins = p_s['n_bins'][best_hyperparams_cte['n_bins']]
    if transformed:
        output.update({'n_bins': n_bins})
    else:
        output.update({'n_bins': None})
    return output
