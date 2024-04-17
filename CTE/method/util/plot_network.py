#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:54:24 2023

@author: xbb
"""


import networkx as nx
from matplotlib import pyplot as plt

def plot_network_start(obj_table: list,
                       parameters: dict,
                       name_group_list: list | None = None,
                       digits: int = 1,
                       colorbar: bool = True) -> None | str:
    '''
    

    Parameters
    ----------
    obj_table : list
        Numerical results of overall feature importance, additive 
        effects, and interaction effects.
    parameters : dict
        For plotting diagrams.
    name_group_list : list | None, optional
        DESCRIPTION. The default is None.
    digits : int, optional
        How many digits of the numerical number to display. 
        The default is 1.
    colorbar : bool, optional
        Whether to present the colorbars. The default is True.

    Returns
    -------
    None
        Plot the network diagram.

    '''

    if name_group_list is None:
        name_group_list = ['X' + str(i + 1) for i in \
                           range(len(obj_table[0]))]
    if len(name_group_list) != len(obj_table[0]):
        return ('length of name_group_list should be equivalent'
                'to the number of nodes')

    base_size = parameters['base_size']
    base_edge_size = parameters['base_edge_size']
    horizontal_positive_shift = \
        parameters['horizontal_positive_shift']
    horizontal_negative_shift = \
        parameters['horizontal_negative_shift']
    vertical_positive_shift = parameters['vertical_positive_shift']
    vertical_negative_shift = parameters['vertical_negative_shift']
    label_font_size = parameters['label_font_size']
    edge_label_font_size = parameters['edge_label_font_size']

    edge_list = []
    ind_edge_list = []
    for i1, name in enumerate(name_group_list):
        for i2, name2 in enumerate(name_group_list):
            # The order of edges (i1 < i2 here) matter.
            # It seems there are bugs in networkx package in the
            # order of drawing edges. See also this stackoverflow
            # question:
            # https://stackoverflow.com/questions/71773541/python-networkx-how-to-draw-graph-with-varying-edge-width
            if i1 < i2:
                edge_list.append((name, name2))
                ind_edge_list.append((name_group_list.index(name),
                                      name_group_list.index(name2)))

    G = nx.DiGraph() # Graph()
    G.add_edges_from(edge_list)

    nodelist = [node for node in G.nodes()]

    # XMDIii / XMDIi * 100 %; additive effect proportions
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    node_size_mdi = obj_table[0] * base_size
    rgba = 1 - obj_table[1]


    pos = nx.circular_layout(G)


    # Draw nodes
    nodes_network = nx.draw_networkx_nodes(
                G,
                pos,
                cmap = 'bwr',
                edgecolors = '#000000',
                vmin = 0,
                vmax = 1,
                node_color = rgba,
                node_size = node_size_mdi)

    # Draw lables
    pos_attrs = {}
    for node, coords in pos.items():
        if coords[0] > 0:
            if coords[1] > 0:
                pos_attrs[node] = (
                    coords[0] - horizontal_positive_shift,
                    coords[1] - vertical_positive_shift)
            else:
                pos_attrs[node] = (
                    coords[0] - horizontal_positive_shift,
                    coords[1] + vertical_negative_shift)
        if coords[0] < 0:
            if coords[1] > 0:
                pos_attrs[node] = (
                    coords[0] + horizontal_negative_shift,
                    coords[1] - vertical_positive_shift)
            else:
                pos_attrs[node] = (
                    coords[0] + horizontal_negative_shift,
                    coords[1] + vertical_negative_shift)

    labels = {}
    index = 0
    for node in G.nodes():
        if obj_table[0][index] == max(obj_table[0]):
            labels[node] = name_group_list[index] + '\n' +  \
                str(round(100 * obj_table[0][index], digits)) + '% '
        else:
            labels[node] = name_group_list[index]
        index = index + 1

    nx.draw_networkx_labels(G,
                            pos_attrs,
                            labels,
                            alpha = 0.7,
                            font_size = label_font_size,
                            font_color ='black',
                            font_family = 'monospace')

    # Draw edges
    width_list = []
    alpha_list = []
    alpha_list_reverse = []
    index = 0
    for _ in edge_list:
        ind_edge = ind_edge_list[index]
        # Ratios of interaction effects to the total sample variance
        total_interaction = obj_table[2][ind_edge[1], ind_edge[0]]
        interaction_proportion = \
            obj_table[3][ind_edge[0], ind_edge[1]]
        interaction_proportion_reverse = \
            obj_table[3][ind_edge[1], ind_edge[0]]

        width_list.append(total_interaction)
        alpha_list.append(interaction_proportion)
        alpha_list_reverse.append(interaction_proportion_reverse)
        index = index + 1
    max_interaction = max(width_list)

    factor_inter = base_edge_size / max_interaction
    width_list = [i * factor_inter for i in width_list]

    nx.draw_networkx_edges(
                G,
                pos,
                connectionstyle = 'arc3, rad=0.1',
                width = width_list,
                edgelist = edge_list,
                edge_color = [(0, 0, 0, q) for q in alpha_list],
                node_size = node_size_mdi,
                nodelist = nodelist,
                arrows = True)

    edge_list_reverse = [(elem[1], elem[0]) for elem in edge_list]
    nx.draw_networkx_edges(
                G,
                pos,
                connectionstyle = 'arc3, rad=0.1',
                width = width_list,
                edgelist = edge_list_reverse,
                edge_color =
                    [(0, 0, 0, q) for q in alpha_list_reverse],
                node_size = node_size_mdi,
                nodelist = nodelist,
                arrows = True)


    # For darwing colorbar in Red/Blue:
    # XMDIij / XMDIi * 100 % or XMDIij / XMDIj * 100 %
    # interaction effect proportions
    if max(alpha_list) >= max(alpha_list_reverse):
        ind_max = alpha_list.index(max(alpha_list))
        nx.draw_networkx_edge_labels(
            G,
            pos,
            font_size = edge_label_font_size,
            bbox = dict(alpha = 0.3, facecolor='none',
                        edgecolor='none'),
            edge_labels = {edge_list[ind_max]: \
                 str(round(100 * max(alpha_list), digits)) + '%'})
    else:
        ind_max = alpha_list_reverse.index(max(alpha_list_reverse))
        nx.draw_networkx_edge_labels(
            G,
            pos,
            font_size = edge_label_font_size,
            bbox = dict(alpha = 0.3, facecolor='none',
                        edgecolor='none'),
            edge_labels = {edge_list_reverse[ind_max]: \
                  str(round(100 * max(alpha_list_reverse), digits))
                  + '%'})

    if colorbar:
        cbar = plt.colorbar(nodes_network, fraction=0.005, pad=-0.03)
        cbar.set_ticks(ticks = [0], labels = [''])
        cbar.ax.tick_params(axis='both', which='major', labelsize=20)
        cbar.ax.tick_params(size=0)

        # For darwing colorbar in Greys:
        nodes_network_2 = nx.draw_networkx_nodes(
                    G,
                    pos,
                    cmap = 'Greys',
                    vmin = 0,
                    vmax = 1,
                    node_color = rgba,
                    node_size = [0 for i in node_size_mdi])

        cbar = plt.colorbar(nodes_network_2,
                            fraction=0.005,
                            pad=-0.02)
        cbar.set_ticks(ticks = [0, 1], labels = ['0%', '100%'])
        cbar.ax.tick_params(axis='both',
                            which='major',
                            labelsize=20)
        cbar.ax.tick_params(size=0)

    plt.box(False)
    plt.show()
    return None
