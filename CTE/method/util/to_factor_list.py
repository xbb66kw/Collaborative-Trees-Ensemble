#!/usr/bins/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:36:05 2023

@author: xbb
"""

import numpy as np
def group_list_fun(size_of_each_group: list) -> list[list[int]]:
    '''
    

    Parameters
    ----------
    size_of_each_group : list        

    Returns
    -------
    list[list[int]]
        Each inner list is a list of feature index in the same group.

    '''
    ind_runner = 0
    ind_group_list = []
    for i in size_of_each_group:
        ind_group_list.append(list(
                    range(ind_runner, ind_runner + i)))
        ind_runner = ind_runner + i
    return ind_group_list

def continuous_to_indicator_group(x: np.ndarray, 
                                  x_test: np.ndarray | None = None, 
                                  n_bins: int = 1) -> np.ndarray:
    '''
    

    Parameters
    ----------
    x : np.ndarray
        A continuous numpy array that may contain NA.
    x_test : np.ndarray | None, optional
        The matrix to be binned. By default, if set to None, 
        x will be assigned to x_test.
    n_bins : int, optional
        Each bin has an equal number of elements for continuous 
        variables. The same values are in the same bin. To group 
        distinct values in a discrete variable into different bins, 
        set bins to be larger than n / by the minimum count of 
        elements that share the same value. The default is 1.

    Returns
    -------
    output_matrix : np.ndarray
        A matrix of a group of indicators. NaN in x will be isolated
        in one column.

    '''
    if x_test is None:
        x_test = x
    output_matrix = np.zeros(len(x_test) * (n_bins + 1))\
        .reshape(len(x_test), n_bins + 1)
    # NaN accounts for one column
    output_matrix[np.isnan(x_test), n_bins] = 1
    x_unique = np.unique(x[~np.isnan(x)])
    for q in range(n_bins - 1):
        index = max(
            int(len(x_unique) * (q + 1) / n_bins) - 1, 0)
        output_matrix[x_test <= np.sort(x_unique)[index], q] = 1
    output_matrix[x_test <= float('Inf'), n_bins - 1] = 1

    for q in range(n_bins - 1):
        i = n_bins - q - 1
        output_matrix[:, i] = output_matrix[:, i] - \
            output_matrix[:, i - 1]
    return output_matrix

def x_to_group_of_indicators(
        X: np.ndarray,
        X_test: np.ndarray | None = None, 
        n_bins: int = 1,
        ind_var: list | None = None, 
        group_list: list | None = None) -> tuple[np.ndarray, list]:
    '''
    

    Parameters
    ----------
    X : np.ndarray
        An input n by p feature matrix.
    X_test : np.ndarray | None, optional
        The matrix to be binned. By default, if set to None, 
        X will be assigned to X_test.
    n_bins : int, optional
        Each bin has an equal number of elements for continuous 
        variables. The same values are in the same bin. To group 
        distinct values in a discrete variable into different bins, 
        set bins to be larger than n / by the minimum count of 
        elements that share the same value. The default is 1.
    ind_var : list | None, optional
        A list of indices of continuous variables. When ind_var is 
        None, nothing is processed here. The default is None.
    group_list : list | None, optional
        A list of lists of solo feature and lists of group of 
        indicators. The default is None, which means all features 
        are solo features. The default is None.

    Returns
    -------
    tuple[np.ndarray, list]
        A transformed numpy feature matrix
        A new group list.

    '''
    if ind_var is None:
        ind_var = []
    if X_test is None:
        X_test = X

    if group_list is None:
        group_list = [[i] for i in range(X.shape[1])]

    new_group_list = []
    counter_ = 0
    x_new = np.empty([X_test.shape[0] * (group_list[-1][-1] + 1)
        * (n_bins + 1)]).reshape(X_test.shape[0],
                        (group_list[-1][-1] + 1) * (n_bins + 1))
    for group in group_list:
        if len(group) == 1 and group[0] in ind_var:
            x_temp = continuous_to_indicator_group(X[:, group[0]],
                x_test = X_test[:, group[0]], n_bins = n_bins)
            x_new[:, (group[0] + counter_):
                  (group[0] + counter_ + x_temp.shape[1])] = x_temp
            new_group_list.append([group[0] + counter_ + i for i in
                                   range(x_temp.shape[1])])
            counter_ = counter_ + x_temp.shape[1] - 1
        else:
            new_group_list.append([counter_ + i for i in group])
            x_temp = X_test[:, group]
            x_new[:, new_group_list[-1]] = x_temp
    x_new = x_new[:, :(new_group_list[-1][-1] + 1)]
    return(x_new, new_group_list)
