#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:11:17 2024

@author: xbb
"""


#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pickle

import numpy as np
from scipy.stats import norm

from method.cte import CollaborativeTreesEnsemble
#%%
####
# run_ = True to run the hyperparameter seraching process
run_ = True
diagnosis_ = False

# save_ = True to save the resutls
save_ = True
re_start = True
# obj_table_list_temp = obj_table_list





#%%
''' if __name__ == '__main__' is required for multiprocessing '''
if __name__ == '__main__' and run_:
    #%%
    # Save the file if needed.
    if re_start:
        obj_table_list = []
    else:
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
        # file = path + 'simulated_data/example_bias_add'
        file = path + 'simulated_data/example_low_bias_add'

        with open(file, 'rb') as f:
            obj_table_list = pickle.load(f)
    R = 100 - len(obj_table_list)
    #%%
    for _ in range(R):
        #%%
        n = 500
        p = 100
        rho = 0.8
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
            X[i, :] = norm.cdf(np.dot(Q[i, :], A))
            # half of the mass is before zero

        # Two examples.
        # This one is in Section 5
        y = 5 * X[:, 0] + 20 * (X[:, 2] - 0.5)**2 + 15 * X[:, 4] + \
            + 2 * X[:, 8] \
            + 10 * np.sin(np.pi * (X[:, 8] - 0.5) * (X[:, 9] - 0.5))

        # An additional example of multiple intearctions
        # y = 2 * X[:, 9] \
        #     + 10 * np.sin(np.pi * (X[:, 1] - 0.5) * (X[:, 9] - 0.5)) \
        #     + 10 * np.sin(np.pi * (X[:, 5] - 0.5) * (X[:, 9] - 0.5)) \
        #     + 10 * np.sin(np.pi * (X[:, 8] - 0.5) * (X[:, 9] - 0.5))

        y = y + np.squeeze(np.random.multivariate_normal(\
                        [0], [[sample_var]], size = y.shape[0]))
         #%%

        # This one is in Section 5
        # y = 5 * X[:, 0] + 20 * (X[:, 2] - 0.5)**2 + 15 * X[:, 4] + \
        #     + 2 * X[:, 8] \
        #     + 10 * np.sin(np.pi * (X[:, 8] - 0.5) * (X[:, 9] - 0.5))
        # with binning
        best_param_1_g = {'n_trees': 12,
         'min_samples_split': 10,
         'min_samples_leaf': 0,
         'random_update': 0.0,
         'alpha': float('Inf'),
         'max_depth': 10,
         'n_bins': 5}

        #%%
        # An additional example of multiple intearctions
        # y = 2 * X[:, 9] \
        #     + 10 * np.sin(np.pi * (X[:, 1] - 0.5) * (X[:, 9] - 0.5)) \
        #     + 10 * np.sin(np.pi * (X[:, 5] - 0.5) * (X[:, 9] - 0.5)) \
        #     + 10 * np.sin(np.pi * (X[:, 8] - 0.5) * (X[:, 9] - 0.5))
        # with binning
        best_param_2_g = {'n_trees': 11,
         'min_samples_split': 20,
         'min_samples_leaf': 10,
         'random_update': 0.001,
         'alpha': 10,
         'max_depth': 10,
         'n_bins': 5}

        #%%
        # Use the one parameter set of interest
        used_param = best_param_1_g
        forest = CollaborativeTreesEnsemble(
            n_estimators = 100,
            dict_param = used_param) #
        #%%
        print('Model training: ', 100 - R + _)
        # Fit the model
        forest.multi_fit(X, y)
        #%%
        #####
        # Call plot_network_start() to plot the network diagram
        # Table object for plotting
        obj_table = forest.diagram_pack
        obj_table_list.append(obj_table)
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
        if save_:
            # Use pickle.dump and pickle.load to save and read lists
            # lambda = 0.8, the example in Section 5.
            # file = path + 'simulated_data/example_bias'
            # lambda = 0.1, the example in Section 5.
            # file = path + 'simulated_data/example_low_bias'
            # lambda = 0.8, additional example.
            # file = path + 'simulated_data/example_bias_add'
            # lambda = 0.1, additional example.
            # file = path + 'simulated_data/example_low_bias_add'

            file = path + 'simulated_data/test_removed_when_ready'
            # To avoid missaving.
            # Uncomment this if you want to save the reuslts.
            # with open(file, 'wb') as f:
            #     pickle.dump(obj_table_list, f)
    #%%
if diagnosis_:
    #%%
    # Get `file` path from the previous code cell.

    # Experiments in Section 5.1
    # Load the obj.
    with open(file, 'rb') as f:
        obj = pickle.load(f)

    ind_insignificant_addi = \
        [l for l in range(100) if not l in [0, 2, 4, 8, 9]]
    significant_list = np.array([0, 2, 4, 8, 9])
    counter_overall = 0
    counter_interaction = 0
    counter_additive = 0
    for r in range(100):
        measures = obj[r][0]


        measures_2 = np.delete(measures, significant_list)
        if min(measures[significant_list]) <= max(measures_2):
            counter_overall = counter_overall + 1

        # Interaction effect between X8 and X9 is strong.
        index = np.where(obj[r][2] == np.max(obj[r][2]))[0]
        if all(index == [8, 9]) or all(index == [9, 8]):
            counter_interaction = counter_interaction + 1

        additive_effect_vector = obj[r][1] * obj[r][0]

        # Additive effects of X0, X2, X4 are strong.
        if min(additive_effect_vector[[0, 2, 4]]) <=\
            max(additive_effect_vector[ind_insignificant_addi]):
            counter_additive = counter_additive + 1
    print(f"Counts of successful interaction separation: "
          f"{counter_interaction}\n"
          f"Counts of failure additivity separation: "
          f"{counter_additive}\n"
          f"Counts of failure overall separation: {counter_overall}")

    #%%
    # Experiment in the Appendix of Section 5.1
    # Load obj
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    # len(obj) # 100 repeated independent experiments

    significant_list = np.array([1, 5, 8, 9])
    significant_interaction = [[8, 9], [5, 9], [1, 9]]
    # bad counts
    counter_overall = 0
    counter_interaction = 0
    for r in range(100):
        measures = obj[r][0]

        measures_2 = np.delete(measures, significant_list)
        if min(measures[significant_list]) <= max(measures_2):
            counter_overall = counter_overall + 1

        # Find out the minimum interaction effect among thos in
        # `significant_interaction.`
        pair_min = float('Inf')
        for pair in significant_interaction:
            if pair_min >= obj[r][2][pair[0], pair[1]]:
                pair_min = obj[r][2][pair[0], pair[1]]


        # Find out the maximum inteaction effects
        # other than those in `significant_interaction.`
        M = np.copy(obj[r][2])
        M[8, 9] = 0
        M[9, 8] = 0
        M[5, 9] = 0
        M[9, 5] = 0
        M[1, 9] = 0
        M[9, 1] = 0
        # M.shape # (100, 100)

        if np.max(M) >= pair_min:
            counter_interaction = counter_interaction + 1
    print(f"Counts of failure interaction separation: "
          f"{counter_interaction}\n"
          f"Counts of failure overall separation: {counter_overall}")
