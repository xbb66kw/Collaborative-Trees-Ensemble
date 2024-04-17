from __future__ import division
from typing import Any
import numpy as np
from scipy.special import softmax
from method.node import Node
from method.util.traverse import traverse_nodes, traverse_nodes_addi
class CollaborativeTreesClassifier():
    def __init__(self, 
                 m_collaborative_trees: int = 6,
                 m_tree_upper_limit: int = 20,
                 min_samples_split: int = 30, 
                 min_samples_leaf: int = 1, 
                 random_update: float = 0.1, 
                 alpha: float = 1, 
                 max_depth: float = 20,
                 bootstrap: bool = True,
                 group_list: list[list[int]] | None = None) -> None:
        '''
        

        Parameters
        ----------
        m_collaborative_trees : int, optional
            The number of decision trees in a set of collaborative 
            trees. The default is 6.
        m_tree_upper_limit : int, optional
            Limitation of the number of trees in a set of 
            collaborative trees. Used for controlling the 
            computation time. The default is 20.
        min_samples_split : int, optional
            New grown nodes with samples less than min_samples_split 
            will not be further split. The default is 30.
        min_samples_leaf : int, optional
            The minimum number of samples needed at a node to
            justify a new node split. The default is 1.
        random_update : float, optional
            The parameter random_update governs the proportion of 
            nodes to be updated in each round, but it does not apply 
            when any root nodes or paired nodes at depth one have 
            not been updated. The default is 0.1.
        alpha : float, optional
            Use probability weights for deciding splits. A low 
            value of alpha makes the weights more uniform. 
            The default is 1.
        max_depth : float, optional
            The maximum number of levels that the tree can grow
            downwards before forcefully becoming a leaf.  
            The default is 20.
        bootstrap : bool, optional
            Whether to randomly chose data to fit each tree on. 
            The default is True.
        group_list : list[list[int]]| None, optional
            A list of lists of indices of feature groups. 
            The default is None.

        Returns
        -------
        None
            Trained m_collaborative_trees collaborative trees.

        '''        
    
        self.m_collaborative_trees = m_collaborative_trees
        #####
        # We set a upper limit for the number of collaborative trees
        self.m_total_tree = min(m_tree_upper_limit,
                                self.m_collaborative_trees)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # self.min_samples_parent = 40
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_update = random_update
        self.alpha = alpha
        
        
        # Matching pursuit parameter
        self.matching_pursuit = True
        # Paired optimization parameter
        self.paired_split = True

        # Private objects
        self.node_v_update: dict[Node, np.ndarray | None] = {}
        self.n_features: int | None = None
        self.group_list = group_list
        self.group_info = []
        if group_list is not None:
            # Some groups have only one feature. These groups are
            # not real groups
            self.group_info = [x for q in group_list
                               if len(q) > 1 for x in q]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        

        Parameters
        ----------
        X : np.ndarray            
        y : np.ndarray

        Returns
        -------
        None
            Creates a forest of decision trees using a random 
            subset of data and features.

        '''
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)
        n_samples = y.shape[0]
        self.n_features = X.shape[1]
        self.min_samples_split = min(self.min_samples_split,
                                     int(n_samples / 10))
        self.k_class = y.shape[1]
        # Set group_list default
        if self.group_list is None:
            self.group_list = [[j] for j in range(X.shape[1])]
        # Boostrap resampling
        if self.bootstrap:
            inds = np.random.choice(a = np.arange(n_samples),
                        size = n_samples, replace = True)
            X = X[inds, :]
            y = y[inds, :]

        # Initialization
        # The list of all nodes
        self.node_v = [Node(X, 
                            None, # feature_index
                            None, # threshold
                            np.arange(n_samples), # ind_set
                            i, # tree_index
                            0, # depth
                            i, # node_index
                            np.zeros(self.k_class) # predicted values
                            )
                        for i in range(self.m_total_tree)]
        
        # Only the nodes with indices in end_node_index would be 
        # considered to be updated
        end_node_index = \
            list(range(int(self.m_collaborative_trees)))
        # Predicted values
        prediction_y_mat = np.zeros(n_samples *
            self.m_total_tree * self.k_class).reshape(
            [self.m_total_tree, n_samples, self.k_class])  

        # Loop until all nodes are sufficiently small or not valid 
        # for further splitting
        while len(end_node_index):
            self.node_v_update = {}            
            # All root notes that are aligible for splitting
            matching_root = [q for q in 
                 end_node_index if self.node_v[q].depth == 0]
            # All nodes at depth 1 that are eligible for splitting
            matching_interaction = [q for q in 
                 end_node_index if self.node_v[q].depth == 1]
            # Update priority
            if self.matching_pursuit and len(matching_root) > 0:
                # Update for root nodes
                for node_index in matching_root:
                    self.node_v_update.\
                        update({self.node_v[node_index] : None})
            elif self.matching_pursuit and \
                len(matching_interaction) > 0:
                for node_index in matching_interaction:
                    self.node_v_update.\
                        update({self.node_v[node_index] : None})
            else:
                # Full/random update:
                # `random_update` works after depth 1.
                num_node = min(max(1,
                    int(len(end_node_index) * self.random_update)),
                                len(end_node_index))                
                for i in np.random.choice(a = end_node_index,
                    size = num_node, replace = False):
                    node = self.node_v[i]
                    self.node_v_update.update({node : None})
                    # Include valid associated nodes for updating
                    for p_node in node.parent.children: 
                        if self.paired_split and p_node.node_index \
                            in end_node_index:                     
                            self.node_v_update.update({
                                p_node : None})
                    
                    
            # `daughter_node_list` may contain daughter nodes for
            # a set of associated nodes to be updated at this round
            node_list, daughter_node_list = \
                self.grow(X, y, prediction_y_mat)
            #####

            # Remove the node that has been chosen to be updated
            # regardless of whether the node has actually been
            # updated            
            for node in node_list:
                end_node_index.remove(node.node_index)

            for daughter_node in daughter_node_list:
                # Valid nodes are appended into `end_node_index`
                if len(daughter_node.ind_set) > \
                    self.min_samples_split and \
                        daughter_node.depth <= self.max_depth\
                            and not daughter_node.stop:
                    end_node_index.append(\
                        daughter_node.node_index)

                # Upate the node vector
                self.node_v.append(daughter_node)
                
                # Update the predicted values
                prediction_y_mat[node.tree_index,
                    daughter_node.ind_set] = daughter_node.ave

    def grow(self, 
             X: np.ndarray,
             y: np.ndarray,
             prediction_y_mat: np.ndarray
             ) -> tuple[list[Node], list[Node]]:
        '''
        

        Parameters
        ----------
        X : np.ndarray
            DESCRIPTION.
        y : np.ndarray
            DESCRIPTION.
        prediction_y_mat : np.ndarray
            DESCRIPTION.

        Returns
        -------
        (tuple[list[Node], list[Node]])
            Grow the potential daughter nodes for the 
            updating nodes.

        '''
        # Calculate the residuals used for updating nodes
        y_residual_0 = y - np.sum(prediction_y_mat, axis = 0)
        y_residual_s = y_residual_0**2
        for key in self.node_v_update:
            self.node_v_update[key]\
                = key.cal_loss(X, 
                               y, 
                               y_residual_0, 
                               y_residual_s,
                               self.group_info)

        ###
        ###
        ###
        # Calculate split scores over each set of (valid) associated 
        # nodes, and each feature group
        node_list = list(self.node_v_update.keys())        
        # `node_loss_list` record the loss gain for each pair of 
        # (a set of associated nodes, feature group)
        node_loss_list: np.ndarray | list[Any] = []
        assert isinstance(node_loss_list, list)
        # `node_info` consists of the corresponding split information
        node_info = []
        for N_ in node_list:
            # Associated nodes are split on the same feature group
            update_set = []
            if N_.parent is None:
                update_set.append(N_)
            else:
                for p_node in N_.parent.children:
                    if self.paired_split and p_node in node_list:
                        update_set.append(p_node)
                        # Remove associated nodes that have been 
                        # considered
                        node_list.remove(p_node)

            # `node_loss_list_temp` loops over each feature group,
            # and compute the respective loss gain
            assert isinstance(self.group_list, list)
            node_loss_list_temp = np.zeros(len(self.group_list))
            for index, group in enumerate(self.group_list):
                split_N: list[tuple[Node, int, Any, float]] = []
                # For each feature group to be split, sum the split
                # scores over the set of associated nodes 
                # (see the big parenthsis in equation (2))
                for node in update_set:
                    node_group_loss_gain = 0
                    for j in group:
                        ind_set = node.ind_set                        
                        temp_ = self.node_v_update[node]
                        assert isinstance(temp_, np.ndarray)
                        j_loss = temp_\
                            [(j * (len(ind_set) - 1)):((j + 1) \
                                    * (len(ind_set) - 1))]
                        ind_best = np.argmax(j_loss)
                        # Loss gain of splitting on this 
                        # feature group. (equation (3) in the paper)
                        node_group_loss_gain = \
                            node_group_loss_gain + \
                                max(j_loss[ind_best], 0)
                    if len(group) > 1:
                        # Split point is `None` for feature groups
                        split_N.append(
                            (node,
                             index,
                             None,
                             node_group_loss_gain)
                            )
                    else:
                        split_N.append(
                            (node,
                             group[0],
                             ind_best,
                             max(j_loss[ind_best], 0))
                            )
                    
                    node_loss_list_temp[index] = \
                        node_loss_list_temp[index] + \
                            max(node_group_loss_gain, 0)
                # `node_loss_list_temp[index]` is 0 when 
                # all the additive loss gain terms are zero.
                # In this case, we set the corresponding split score
                # of the set of assocaited nodes to be -float('Inf').
                if node_loss_list_temp[index] == 0:
                    node_loss_list_temp[index] = -float('Inf')
                node_info.append(split_N)
            node_loss_list.extend(node_loss_list_temp)

        
        # Optimization (equation (2) in the paper)
        node_loss_list = np.array(node_loss_list)
        ind_best_ = optimization(node_loss_list, self.alpha)

        # Grow all the daughter nodes for all updating 
        # associated nodes
        counter_ = 0 # count the number of new nodes
        node_update_list = []
        # Record the potential daughter nodes for the 
        # updating nodes.
        node_daughter_list = []   
        for elem in node_info[ind_best_]:
            # `elem`: (node, group index or solo feature index, 
            # None or split coordinate, split score)
            
            # The node that will be updated, and removed
            # from the update waiting list
            node = elem[0]
            node_update_list.append(node)
            # Release the memory
            del node.X_arg_temp
            del node.seq1
            del node.seq2
            
            loss_gain = elem[3]
            
            # `update_ind_list` is a list of indices of the subsample
            # of each child node of `elem[0]`.
            update_ind_list = []
            ind_set = node.ind_set
            if elem[2] is not None:
                j = elem[1]
                X_j = X[ind_set, :][:, j]
                sorted_indices = np.argsort(X_j)
                i = elem[2]
                threshold = (X_j[sorted_indices[i]]\
                              + X_j[sorted_indices[i+1]]) / 2

                ind_subset = X_j <= threshold
                # Xj with smaller values first
                update_ind_list.append(ind_subset)
                update_ind_list.append(~ind_subset)
                feature_index: int | list[int] = j
            else:
                # A list of features
                assert isinstance(self.group_list, list)
                feature_index = self.group_list[elem[1]]
                # Features in groups have values of one or zero
                # Since the value of threshold is given as 0.5,
                # the update only applies to indicator feature groups
                threshold = 0.5
                for j in feature_index:
                    X_j = X[ind_set, :][:, j]    
                    update_ind_list.append(~ (X_j <= threshold))

         
            #######
            # Continue to split if the node is larger than 40
            # and that one of the daughter nodes is big enough
            continue_grow = False
            continue_grow_counter = 0
            for ind_subset in update_ind_list:
                if sum(ind_subset) > self.min_samples_leaf:
                    continue_grow_counter = continue_grow_counter + 1
            
            if continue_grow_counter >= 2:
                continue_grow = True


            ###
            ### --- Grow daughter nodes ---
            # 1. Residuals coditional on `ind_set`.
            # 2. The way we update the hat{k}th tree is described
            #   in the paper. Note that here we subtract the 
            #   hat{k}th tree prediction first.
            # 3. `y_hat_temp_all[ind_update_tree]` below is a
            #   constant vector.
            ind_update_tree = node.tree_index
            y_hat_temp_all = prediction_y_mat[:, ind_set]
            y_residual = y_residual_0[ind_set]
            y_temp = y_residual + \
                y_hat_temp_all[ind_update_tree]
            
            
            # Calculate and record the node importance
            node.importance = loss_gain / X.shape[0]

            # The value of `node.threshold` is given here. 
            # Therefore, `node.threshold` is None implies 
            # the node has no daughter nodes.
            node.threshold = threshold
            node.feature_index = feature_index

            for ind_subset in update_ind_list:
                # Small nodes do not continue to grow
                if sum(ind_subset) <= self.min_samples_leaf:                  
                    false_ave = \
                        y_hat_temp_all[ind_update_tree][0, 0]
                    child_node = Node(
                        X,
                        None, # feature_index
                        None, # threshold
                        node.ind_set[ind_subset], # subsample indices
                        node.tree_index, # tree index
                        node.depth + 1, # depth
                        len(self.node_v) + counter_, # node index
                        false_ave # predicted values
                        )
                    counter_ = counter_ + 1
                    child_node.parent = node
                    child_node.stop = True
                else:
                    # Update the node split information and 
                    # daughter node    
                    false_ave = np.mean(y_temp[ind_subset],
                                        axis = 0)[0]
                    child_node = Node(
                        X,
                        None, # feature_index
                        None, # threshold
                        node.ind_set[ind_subset], # subsample indices
                        node.tree_index, # tree index
                        node.depth + 1, # depth
                        len(self.node_v) + counter_, # node index
                        false_ave # predicted values
                        )
                    counter_ = counter_ + 1
                    child_node.parent = node
                    if not continue_grow:
                        child_node.stop = True
                    
                # Append to node's child_node list
                # The children's order is the same as 
                # the group's order
                node.children.append(child_node)
                # Append to daughter node list                                         
                node_daughter_list.append(child_node)
                
        return node_update_list, node_daughter_list


    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        

        Parameters
        ----------
        X : np.ndarray            

        Returns
        -------
        np.ndarray
            Predict the value of each sample in X.

        '''
        predictions = np.empty([self.m_collaborative_trees,
                                X.shape[0], 
                                self.k_class])
        for i in range(self.m_collaborative_trees):
            predictions[i] = self.node_v[i].predict(X)
        return np.sum(predictions, axis = 0)

    def importance(
            self, 
            group_list: list[list[int]] | None = None) -> None:
        '''
        

        Parameters
        ----------
        group_list : list[list[int]] | None, optional
            Information of feature groups. The default is None.

        Returns
        -------
        None
            Calculate the importance matrix.

        '''
        if group_list is None:
            B = self.n_features
        else:
            B = len(group_list)

        assert isinstance(B, int)
        self.mdi_total = np.zeros(B)
        self.importance_matrix = np.zeros(B**2)\
            .reshape(B, B)
        for i in range(self.m_total_tree):
            traverse_nodes(self.importance_matrix,
                           self.mdi_total, 
                           self.node_v[i], 
                           None, 
                           None, 
                           group_list)

        self.importance_addi = np.zeros(B)
        for i in range(self.m_total_tree):
            traverse_nodes_addi(self.importance_addi,
                                self.node_v[i], 
                                group_list)

def optimization(loss_v: np.ndarray, alpha: float) -> int:
    if alpha != float('Inf'):
        # An edge case where weights of all splits are
        # float('Inf')
        if all([q == -float('Inf') for q in loss_v]):
            ind_best = np.random.choice(a =
                    np.arange(len(loss_v)))
        else:
            # Inspired by BART
            loss_v_temp = softmax(alpha \
                * loss_v)
            ind_best = np.random.choice(a =
                    np.arange(len(loss_v)),
                    p = loss_v_temp)
    else:
        # ties are borken randomly
        ind_best = np.random.choice(np.flatnonzero(\
            loss_v == loss_v.max()))
    return ind_best
