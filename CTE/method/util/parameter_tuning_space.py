#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:45:07 2023

@author: xbb
"""
from typing import Any, Dict
import numpy as np
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, hp
from sklearn.metrics import mean_squared_error
from hyperopt.pyll.base import scope
# Useful when debugging
import hyperopt.pyll.stochastic
# print(hyperopt.pyll.stochastic.sample(space))
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0

# from method.util.to_factor_list import x_to_group_of_indicators
from method.cte import CollaborativeTreesEnsemble




#%%
# Tuning parameter space and objective functions for XGBoost

space_xgb = {
    'max_depth': scope.int(hp.quniform("max_depth", 2, 15, 1)),
    'gamma': hp.uniform('gamma', np.log(1e-8), np.log(7)),
    'reg_alpha': hp.uniform('reg_alpha', np.log(1e-8), np.log(1e2)),
    'reg_lambda': hp.uniform('reg_lambda', np.log(0.8), np.log(4)),
    'learning_rate': hp.uniform('learning_rate', np.log(1e-5), np.log(0.7)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
    'n_estimators': 1000}

# print(hyperopt.pyll.stochastic.sample(space_xgb))
# test_set = hyperopt.pyll.stochastic.sample(space_xgb)
# print(np.exp(test_set['gamma']), np.exp(test_set['learning_rate']), np.exp(test_set['reg_alpha']), np.exp(test_set['reg_lambda']), test_set['min_child_weight'])


def objective_xgb_regression(space):
    X = space['X']
    y = space['y']
    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.
    X_train_in, X_train_out, y_train_in, y_train_out \
        = train_test_split(X, y, test_size = 0.4)
        
    
    model=xgb.XGBRegressor(
        n_estimators =space['n_estimators'], 
        max_depth = int(space['max_depth']), 
        gamma = np.exp(space['gamma']),
        reg_alpha = np.exp(space['reg_alpha']),
        reg_lambda = np.exp(space['reg_lambda']),
        learning_rate = np.exp(space['learning_rate']),
        min_child_weight=space['min_child_weight'],
        colsample_bytree=space['colsample_bytree'],
        colsample_bylevel = space['colsample_bylevel'],
        subsample = space['subsample'])
   

    
    # Define evaluation datasets.
    evaluation = [( X_train_in, y_train_in), 
                  ( X_train_out, y_train_out)]
    
    # Fit the model. Define evaluation sets, early_stopping_rounds,
    # and eval_metric.
    model.fit(X_train_in, y_train_in,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=20,verbose=False)

    # Obtain prediction and rmse score.
    pred = model.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, pred)
    
    # Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}

#%%
#####
# Tuning parameter space and objective functions for CTE.
p_s: Dict[str, Any] = {'n_bins': [5, 7, 10, 15, 20, 40],
       'min_samples_split': [5, 10, 15, 20, 30],
       'min_samples_leaf': [0, 5, 10, 15, 20, 30],
       'n_trees': [q + 1 for q in range(5, 12)],
       'random_update': [0.0, 0.0001, 0.001, 0.01, 0.1, 1],
       'alpha': [0.001, 1, 10, 100, 1E+4, float('Inf')],
       'max_depth': [3, 5, 10, 20, 30, float('Inf')]}

space_cte = {
    'n_bins': scope.int(hp.choice('n_bins',
                                p_s['n_bins'])),
    'min_samples_split': scope.int(hp.choice('min_samples_split',
                                p_s['min_samples_split'])),
    'min_samples_leaf': scope.int(hp.choice('min_samples_leaf',
                                p_s['min_samples_leaf'])),
    'random_update': hp.choice('random_update', 
                               p_s['random_update']),
    'n_trees': hp.choice('n_trees', p_s['n_trees']),
    'alpha': hp.choice('alpha', p_s['alpha']),
    'max_depth': hp.choice('max_depth', p_s['max_depth'])}



def objective_cte_regression(space: dict) -> dict:
    '''
    

    Parameters
    ----------
    space : dict
        Things needed for optimizations.

    Returns
    -------
    dict
        Optimized hyperparameters. The reported values may be the 
        indices of the optimizaed values from their respective 
        hyperparameter space.

    '''
    # Specify the number of trees in CTE if you want.
    if space['n_trees_user'] is not None:
        space['n_trees'] = space['n_trees_user']

    group_list = space['group_list']
    if group_list is not None:
        # Seems to be a bug of `fmin` function
        group_list = [list(q) for q in space['group_list']]
    X = space['X']
    y = space['y']

    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.
    X_train_in, X_train_out, y_train_in, y_train_out \
        = train_test_split(X, y, test_size = 0.4)


    
    if space['transformed']:
        n_bins = space['n_bins']
    else:
        # Set `n_bins = None` means that no additional feature
        # binning is used during model training.
        n_bins = None
    # The parameters obtained from space['...'] are their real values
    # E.g., n_trees = 1 means there is only one tree in the CTE
    clf = CollaborativeTreesEnsemble(n_estimators = 30,
            m_collaborative_trees = space['n_trees'],
            random_update = space['random_update'],
            min_samples_split = int(space['min_samples_split']),
            min_samples_leaf = int(space['min_samples_leaf']),
            alpha = space['alpha'],
            max_depth = space['max_depth'], 
            n_bins = n_bins)
    
    if space['multiprocessing'] > 1:
        clf.multi_fit(X_train_in, 
                      y_train_in, 
                      group_list = group_list, 
                      num_cpu = space['multiprocessing'])
    else:
        clf.fit(X_train_in, y_train_in, 
                      group_list = group_list)
    prediction = clf.predict(X_train_out)
    mse = mean_squared_error(y_train_out, prediction)
    
    return {'loss': mse, 'status': STATUS_OK }


#%%
p_s_rf = {'max_depth': [None, 5, 10, 20, 50],
          'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
          'criterion': ['squared_error', 'absolute_error']}

space_rf = {
    'gamma': hp.uniform('gamma', 0, 1), 
    'max_depth': hp.choice('max_depth', p_s_rf['max_depth']),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', p_s_rf['min_impurity_decrease']),
    'criterion': hp.choice('criterion', p_s_rf['criterion'])}


def objective_rf_regression(space):
    X = space['X']
    y = space['y']
    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.
    X_train_in, X_train_out, y_train_in, y_train_out \
        = train_test_split(X, y, test_size = 0.4)
        
    clf = RandomForestRegressor(max_depth = space['max_depth'],
                max_features = space['gamma'],
                min_samples_leaf = int(space['min_samples_leaf']),
                min_samples_split = int(space['min_samples_split']),
                criterion = space['criterion'],
                min_impurity_decrease = space['min_impurity_decrease'])
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}


