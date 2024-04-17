#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:56:34 2023

@author: xbb
"""
import numpy as np
from method.node import Node

def index_group(j: int | list[int], group_list: list[list[int]]) -> int | None:
    '''
    

    Parameters
    ----------
    j : int | list[int]
        Index of the feature of interest, or a list of indices of
        features of interest
    group_list : list[list[int]]
        List of groups. A solo feature is seen as a group having 
        only a feature.

    Returns
    -------
    int | None
        The index of the group having j.

    '''
    index = 0
    for g in group_list:
        if j in g or j == g:
            return index
        index = index + 1
    return None

def traverse_nodes(importance_matrix: np.ndarray,
                   mdi_total: np.ndarray,
                   node: Node,
                   anchor_feature: int | None,
                   green_flag_feature: int | None,
                   group_list: list[list[int]] | None = None) -> None:
    '''
    

    Parameters
    ----------
    importance_matrix : np.ndarray
        p by p importance matrix. The diagnal entries are zeros.
    mdi_total : np.ndarray
        The overall importance of each feature.
    node : Node
        A Node recording the importance and its decendants.
    anchor_feature : int | None
        For recursive calculation.
    green_flag_feature : int | None
        For recursive calculation.
    group_list : list[list[int]] | None, optional
        If groups are considred, their importance should be 
        calculated accordingly. The default is None.

    Returns
    -------
    None
        No return. The ouput is `importance_matrix` and `mid_total.`

    '''
    # Each Rows of the matrix indicate the feature to be split and
    # contribute the MDI (and exMDI)
    if group_list is None:
        group_list = [[l] for l in range(importance_matrix.shape[0])]

    # j may be a list of indices of a group of indicator features
    j = node.feature_index
    # Variable 'j' must be an integer or a list of integers.
    assert isinstance(j, int | list)
    ind_j = index_group(j, group_list)
    # MDI style feature importance for collaborative trees
    mdi_total[ind_j] = mdi_total[ind_j] + node.importance

    # Do not record the importance due to additive effects since
    # here we are considering interaction effects.
    if green_flag_feature is None and anchor_feature is None:
        # Starting point
        green_flag_feature = ind_j
    elif green_flag_feature == ind_j and anchor_feature is not None:
        importance_matrix[green_flag_feature, anchor_feature] = \
          importance_matrix[green_flag_feature, anchor_feature]\
             + node.importance
    elif green_flag_feature != ind_j:
        # anchor_feature records the feature at the last
        # switching point
        anchor_feature = green_flag_feature
        green_flag_feature = ind_j

        importance_matrix[green_flag_feature, anchor_feature] = \
          importance_matrix[green_flag_feature, anchor_feature]\
             + node.importance


    # Recusive calculation
    # if node.children is not []:
    for child_node in node.children:
        if child_node.feature_index is not None:
            traverse_nodes(importance_matrix, 
                           mdi_total,
                           child_node, 
                           anchor_feature, 
                           green_flag_feature,
                           group_list)

def traverse_nodes_addi(
        vector: np.ndarray, 
        node: Node, 
        group_list: list[list[int]] | None = None) -> None:
    '''
    

    Parameters
    ----------
    vector : np.ndarray
        The importance of additive effect of each feature.
    node : Node
        A Node recording the importance and its decendants.
    group_list : list[list[int]] | None, optional
        If groups are considred, their importance should be 
        calculated accordingly. The default is None.

    Returns
    -------
    None
        No return. The ouput is `vector.`

    '''
    if group_list is None:
        group_list = [[l] for l in range(len(vector))]
    # j may be a list of indices of a group of indicator features
    j = node.feature_index
    # Variable 'j' must be an integer or a list of integers.
    assert isinstance(j, int | list)
    ind_j = index_group(j, group_list)
    vector[ind_j] = vector[ind_j] + node.importance

    if node.children != []:
        for child_node in node.children:
            if child_node.feature_index is not None:
                ind_true_j = \
                    index_group(child_node.feature_index, group_list)            
                if ind_true_j == ind_j:
                    traverse_nodes_addi(vector, 
                                        child_node,
                                        group_list)
