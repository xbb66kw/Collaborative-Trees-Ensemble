#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:14:06 2023

@author: xbb
"""
from datetime import datetime
import warnings, os, pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np

from sklearn.model_selection import train_test_split
# import packages for hyperparameters tuning
from hyperopt import Trials, fmin, tpe
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
# xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0

# import openml

from method.cte import CollaborativeTreesEnsemble
from method.util.parameter_tuning_space import space_xgb, \
    space_rf, objective_xgb_regression, p_s_rf,\
    objective_rf_regression
from method.util.param_search import search_start


#%%
#####
# Initialization
run_cte = True
run_xgb = True
run_rf = True
run_ls = True
# save_ = True to save results
save_ = True


# Do not touch these parameters
max_evals = 100
test_size = 0.2
validation_size = 0.4

# Repeat R times. Default R = 10
R = 10



# To avoid some weird bugs due to multiprocessing.
start_ = True
if start_:
    # Save the r-square scores for each model
    obj_rsquare_score = []
    for q in range(R):
        obj_rsquare_score.append([])
    start_ = False


#%%
''' if __name__ == '__main__' is required for multiprocessing '''
if __name__ == '__main__':
    path_temp = os.getcwd()
    result = path_temp.split("/")
    path = ''
    checker = True
    for elem in result:
        if elem != 'Collaborative-Trees-Ensemble' and checker:
            path = path + elem + '/'
        else:
            checker = False
    path = path + 'Collaborative-Trees-Ensemble' + '/'

    for ind_sample in range(19):
        for q in range(R):
            
            file = path + 'openml/dataset/' + str(ind_sample)
            with open(file, 'rb') as f:
                dataset = pickle.load(f)
            X, y, categorical_indicator, attribute_names \
                = dataset.get_data(dataset_format = "dataframe", 
                target = dataset.default_target_attribute)
            
            X = X.astype(np.float64)
            y = y.astype(np.float64)
            
            ind_rand = np.random.choice(np.arange(len(y)), 
                        size = min(10000, len(y)), replace = False)
            X = np.array(X)[ind_rand, :]
            y = np.array(y)[ind_rand]
            
            
            
            X_train, X_test, y_train, y_test\
                = train_test_split(X, y, test_size=test_size)
            X_test.shape

            print(X_train.shape)
            print(dataset)
            
            X_train_in, X_train_out, y_train_in, y_train_out \
                = train_test_split(X_train, y_train, 
                                   test_size = validation_size)
        
            #####
            # run CTE
            if run_cte:
                
                best_param = search_start(X_train, 
                                          y_train,
                                          max_evals = max_evals, 
                                          transformed = False,
                                          multiprocessing = 20)
                forest = CollaborativeTreesEnsemble(
                    n_estimators = 100, dict_param = best_param) #
            
                print('Best indices for tuning parameters for CTE: ',
                      forest.get_info())
                print('a round starts', datetime.now().minute,
                      datetime.now().second) 
    
                forest.multi_fit(X_train, y_train)
                print('the round ends', datetime.now().minute,
                      datetime.now().second)
                
                # For testing
                print('Used parameter configuration for CTE: ', 
                      forest.get_info())
                # the lower the better
                print('CTE: ', mean_squared_error(forest.predict(X_test), y_test) / np.var(y_test))
            
                
        
            #####
            # run XGBoost
            if run_xgb:                
                space_xgb['X'] = X_train
                space_xgb['y'] = y_train
                    
                trials = Trials()
                
                
                best_hyperparams = fmin(fn = objective_xgb_regression,
                            space = space_xgb,
                            algo = tpe.suggest,
                            max_evals = max_evals,
                            trials = trials)


                xgbc = xgb.XGBRegressor(n_estimators = 1000, 
                    max_depth = int(best_hyperparams['max_depth']),
                    gamma = np.exp(best_hyperparams['gamma']),
                    reg_alpha = np.exp(best_hyperparams['reg_alpha']),
                    reg_lambda = np.exp(best_hyperparams['reg_lambda']),
                    min_child_weight = int(best_hyperparams['min_child_weight']),
                    colsample_bytree = best_hyperparams['colsample_bytree'],
                    colsample_bylevel = best_hyperparams['colsample_bylevel'],
                    subsample = best_hyperparams['subsample'],
                    learning_rate = np.exp(best_hyperparams['learning_rate']))
                    
            
                xgbc.fit(X_train, y_train)
                print('XGB: ', mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test))
       
            #####
            # run Random Forests
            if run_rf:
                space_rf['X'] = X_train
                space_rf['y'] = y_train
                    
                trials = Trials()
            
                
                best_hyperparams_rf = fmin(fn = objective_rf_regression,
                            space = space_rf,
                            algo = tpe.suggest,
                            max_evals = max_evals,
                            trials = trials)
            
            
                print(best_hyperparams_rf['criterion'], p_s_rf['min_impurity_decrease'][best_hyperparams_rf['min_impurity_decrease']], 'test')
                rf = RandomForestRegressor(n_estimators = 500,
                    max_depth = 
                        p_s_rf['max_depth'][best_hyperparams_rf['max_depth']],
                    max_features = best_hyperparams_rf['gamma'],
                    min_samples_leaf = int(best_hyperparams_rf['min_samples_leaf']),
                    min_samples_split = int(best_hyperparams_rf['min_samples_split']),
                    min_impurity_decrease = p_s_rf['min_impurity_decrease'][best_hyperparams_rf['min_impurity_decrease']],
                    criterion = p_s_rf['criterion'][best_hyperparams_rf['criterion']],)
                rf.fit(X_train, y_train)
            
                print('RF: ', mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test))
            #####
            # Run Least-Square regression
            if run_ls:
                reg = LinearRegression().fit(X_train, y_train)
                reg.score(X_test, y_test)
            #####
            # Save file
            if save_:
                #####
                # Save the r square results for all models
                # obj_rsquare_score is a list of R dicts of 
                # r-square scores of each model
                if len(obj_rsquare_score[q]) > ind_sample:
                    obj_rsquare_score[q] = {
                        'cte': 1 - mean_squared_error(forest.predict(X_test), y_test) / np.var(y_test),
                        'xgb': 1 - mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test),
                        'rf': 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test),
                        'ls': reg.score(X_test, y_test),
                        'dataset': dataset.name}
                else:
                    obj_rsquare_score[q].append({
                        'cte': 1 - mean_squared_error(forest.predict(X_test), y_test) / np.var(y_test),
                        'xgb': 1 - mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test),
                        'rf': 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test),
                        'ls': reg.score(X_test, y_test),
                        'dataset': dataset.name})
                    

                file = path + 'openml/results/openml_rsquare_removed_when_ready'
                print('Results for thebest parameters for openml are \
                      saving at: ', '\n', file)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_score, f)
              
        
        