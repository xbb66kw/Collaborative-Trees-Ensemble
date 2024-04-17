#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:36:35 2023

@author: xbb
"""
import warnings, os
# os.cpu_count()
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pyreadr, pickle


from method.cte import CollaborativeTreesEnsemble
from method.util.param_search import search_start
from method.util.to_factor_list import group_list_fun


from method.util.plot_network import plot_network_start

####
# run_ = True to run the hyperparameter seraching process
run_ = False

# save_ = True to save the resutls
save_ = False


# number of iteration for validation
max_evals = 100
#%%
''' if __name__ == '__main__' is required for multiprocessing '''
if __name__ == '__main__':    
#%%
    # Get the directory path for loading 
    # data_process_embryogrowth.rds
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
    file = path + 'embryogrowth/data_process_embryogrowth.rds'
    
    
#%%
    
    # Python reads the .rds file
    result = pyreadr.read_r(file)
    data = np.array(result[None])
    X = data[:, :(data.shape[1] - 1)]
    y = data[:, -1]

    # Set up the group list
    name_group_list = ['species', 'subspecies', 'area', 'RMU', 'amplitude', 
                       'incubation periods (days)', 'temperature']
    size_of_each_group = [60, 10, 81, 14, 1, 1, 1]
    group_list = group_list_fun(size_of_each_group)
        
#%%
    if run_:
        #%%
        # 1. Hyperparameter tuning process.
        # 2. Used for CTE object dict_param.
        # 3. The information of groups in `group_list` is used here:
        #   `group_list` provides the basic information of groups, 
        #   and `transformed` is True, so all solo features in
        #   `group_list` are transformed into gorups, with `n_bins`
        #   as one of the hyperparameters.
        # 4. There are runtime warnings due to NaN values in X. 
        #   To address this issue, you can fix it by setting 
        #   X[np.isnan(X)] to other values.
        best_hyperparams_cte = search_start(X, y, 
            max_evals = max_evals, group_list = group_list,
            transformed = True)
        #%%
        # The saved best parameter configuration from
        # `best_hyperparams_cte`.
        best_param = {'n_trees': 11,
         'min_samples_split': 5,
         'min_samples_leaf': 0,
         'random_update': 0.01,
         'alpha': 1,
         'max_depth': 5,
         'n_bins': 40}
        
        #%%
        # Train the CTE model
        used_param = best_param
        forest = CollaborativeTreesEnsemble(n_estimators = 100,
                                dict_param = used_param)
        forest.get_info()
        #%%
        # To use group information, groups in `X` must match 
        # those groups specified in `group_list_new`.
        # Since `n_bins` is not None here, all solo features 
        # in `gorup_list` are transformned into groups 
        # (with `n_bins` bins)
        forest.multi_fit(X, y, group_list = group_list)

        
        #%%
        # Table object `obj` for drawing network diagram.
        obj = forest.diagram_pack
        #%%
        # Change these bases to make figures fit better
        # base of node sizes
        base_size = 15000
        # base of edges' sizes
        base_edge_size = 10
        # Positiions of labels
        horizontal_positive_shift = 0.17
        horizontal_negative_shift = 0.15
        vertical_negative_shift = 0.25
        vertical_positive_shift = 0.10
        #
        label_font_size = 30
        parameters = {}
        parameters['base_size'] = base_size
        parameters['base_edge_size'] = base_edge_size
        parameters['horizontal_positive_shift'] = horizontal_positive_shift
        parameters['horizontal_negative_shift'] = horizontal_negative_shift
        parameters['vertical_positive_shift'] = vertical_positive_shift
        parameters['vertical_negative_shift'] = vertical_negative_shift
        parameters['label_font_size'] = label_font_size
        parameters['edge_label_font_size'] = 50
        
        # Draw the network diagram
        # Save the figure of embryogrowth.png from the python 
        # figure window
        plot_network_start(obj, parameters,
                   name_group_list = name_group_list)
        #%%
        if save_:
            # Use pickle.dump and pickle.load to save and read lists
            file = path + 'embryogrowth/results/embryogrowth_results'
            
            # To avoid missaving
            # with open(file, 'wb') as f:
            #     pickle.dump(obj, f)
            
            # Loading the obj from forest.diagram_pack
            with open(file, 'rb') as f:
                obj = pickle.load(f)
            