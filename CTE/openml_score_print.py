#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:45:57 2023

@author: xbb
"""

import os, pickle
import numpy as np

#%%


dataset_name = {0: "cpu_act",
                1: "pol",
                2: "elevators",
                3: "wine_quality",
                4: "Ailerons",
                5: "houses",
                6: "house_16H",
                7: "diamonds",
                8: "Brazilian_houses",
                9: "Bike_Sharing_Demand",
                10: "nyc-taxi-green-dec-2016",
                11: "house_sales",
                12: "sulfur",
                13: "medical_charges",
                14: "MiamiHousing2016",
                15: "superconduct",
                16: "yprop_4_1",
                17: "abalone",
                18: "delay_zurich_transport"}

#%%

# Get the directory path for loading data_process_embryogrowth.rds
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
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.

#####
# Manually control for outputing summary results
# Codes include file reading commends

file = path + 'openml/results/openml_rsquare'
# file = path + 'openml/results/openml_rsquare_removed_when_ready'
with open(file, 'rb') as f:
    obj_rsquare_score = pickle.load(f)
#%%
#####
# obj_rsquare_score is a list of length 10. Each records the 
# R^2 scores for all four methods (including the linear 
# regression) on each of the 19 datasets.
# See obj_rsquare_score[j], j = 0, ..., 18 for details.
if False:
    #%%
    #####
    # Across datasets comparison
    R = 10  # number of repetition in the numerical experiments
    D_ = 10
    # Method; Dataset; Repetition
    result_table = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['cte'],
                                       results[j]['xgb'], results[j]['rf']]

    score_all = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            M = np.max(result_table[:, j, ind])
            m = np.min(result_table[:, j, ind])
            for method in range(3):
                # Win rates
                score_all[method, j, ind] = \
                    (result_table[method, j, ind] - m) / (M - m)

    # Print the overall results
    # method = 0 (cte), 1 (xgb), 2 (rf)
    method = 2
    print('average winning rate over ' + str(R) + ' experiemnts for \
          method ' + str(method) + ': ',
          np.mean(score_all[method]), '\n',
          'max wining rate: ',
          np.max(np.mean(score_all[method], axis=0)), '\n',
          'min wining rate: ',
          np.min(np.mean(score_all[method], axis=0)))

    #%%

    # Method; Dataset; Repetition
    result_table = np.zeros(4 * D_ * R).reshape(4, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['cte'],
                                       results[j]['xgb'], results[j]['rf'],
                                       results[j]['ls']]

    # Print the results
    # ind_dataset = 0, ..., 18
    ind_dataset = 18
    print('The R^2 scores for all three methods (cte, xgb, rf) based\
          on the ' + str(ind_dataset) + 'th dataset: ',
          dataset_name[ind_dataset])

    print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
