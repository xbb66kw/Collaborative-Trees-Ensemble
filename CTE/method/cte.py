from __future__ import division
from multiprocessing import Pool
import numpy as np
from method.collaborativetree import CollaborativeTreesClassifier
from method.util.to_factor_list import x_to_group_of_indicators
class CollaborativeTreesEnsemble():
    def __init__(self,
                 n_estimators: int = 100,
                 m_collaborative_trees: int = 6,
                 random_update: float = 0.1,
                 min_samples_split: int = 1,
                 min_samples_leaf: int = 5,
                 alpha: float = 1.0,
                 n_bins: int | None = None,
                 max_depth: float = float('Inf'),
                 dict_param: dict | None = None) -> None:
        self.n_estimators = n_estimators
        self.forests = []
        self.X = None
        self.n_bins = n_bins
        #####
        # If `dict_param` is provided, then input all parameters from
        # `dict_param`.
        param_list =['n_trees',
                     'min_samples_split',
                     'min_samples_leaf',
                     'random_update',
                     'alpha',
                     'max_depth',
                     'n_bins']
        if type(dict_param) is dict and \
            all(key in dict_param.keys() for key in param_list):
            self.m_collaborative_trees = dict_param['n_trees']
            self.min_samples_split = dict_param['min_samples_split']
            self.min_samples_leaf = dict_param['min_samples_leaf']
            self.random_update = dict_param['random_update']
            self.alpha = dict_param['alpha']
            self.max_depth = dict_param['max_depth']
            self.n_bins = dict_param['n_bins']
        else:
            self.m_collaborative_trees = max(1, m_collaborative_trees)
            # Minimum sample size required for further spliting
            self.min_samples_split = min_samples_split
            # Minimum sample size required for all end nodes
            self.min_samples_leaf = min_samples_leaf
            # Size of the subset of random candidate nodes
            # for updating. Proportion of all nodes.
            self.random_update = random_update
            # Probability weights factor.
            self.alpha = alpha
            self.max_depth = max_depth
    def get_info(self):
        return {'n_trees': self.m_collaborative_trees,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_update': self.random_update,
                'alpha': self.alpha,
                'max_depth': self.max_depth,
                'n_bins': self.n_bins}

    def multi_fit(self, 
                  X: np.ndarray,
                  y: np.ndarray,
                  group_list: list[list[int]] | None = None, 
                  num_cpu: int = 20) -> None:
        '''
        

        Parameters
        ----------
        X : np.ndarray

        y : np.ndarray

        group_list : list[list[int]] | None, optional
            A list of lists of indices of feature groups. 
            The default is None.
            
        num_cpu : int
            The number of CPUs used for multiprocessing.
            
        Returns
        -------
        None
            Fit a Collaborative Trees Ensemble model with 
            multiprocessing. Calculate the feature importance.

        '''
        # `if __name__ == '__main__'` is required for multiprocessing
        self.y_mean = np.mean(y, axis = 0)
        y = y - np.mean(y, axis = 0)
        self.y = y
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)
        
        
        # If self.n_bins is not None, then all solo features are
        # transformed into groups of one-hot indicators
        if self.n_bins is not None:
            self.X = X
            if group_list is None:
                # Indices of all continuous variables that will be
                # transformed into groups
                self.ind_var = [i for i in range(X.shape[1])]
                self.group_list_ori = \
                    [[i] for i in range(X.shape[1])]
            else:
                # Extract indices of all continuous variables                
                self.ind_var = [g[0] for g in group_list 
                                if len(g) == 1]
                self.group_list_ori = group_list
            X, group_list = data_transform(X,
                                           None,
                                           self.n_bins, 
                                           self.ind_var, 
                                           group_list)
        self.group_list = group_list
        self.n_features = X.shape[1]
        self.k_class = y.shape[1]
        self.forests = []

        ########
        if not group_list is None:
            checker = isinstance(group_list, list)
            if checker:
                for elem in group_list:
                    if not isinstance(elem, list):
                        checker = False
            if checker:
                list_temp = [y for elem in group_list for y in elem]
                if list_temp != [l for l in range(self.n_features)]:
                    return('group_list should consist of column'
                           'indices that are collectively a'
                               'partition of all column indices')
            else:
                return('group_list should be list of lists, where'
                        'each inner list consists of column indices'
                        'and all inner lists colletively are a'
                        'partition of column indices')
        if not group_list is None:
            for group in [x for x in group_list if len(x) > 1]:
                total_ = np.sum(X[:, group] == 0) \
                    + np.sum(X[:, group] == 1)
                for i in range(X.shape[0]):
                    if np.sum(X[i, group]) != 1 \
                        or total_ != X.shape[0] * len(group):
                        return('Variables in your groups may not be'
                               'one-hot indicators')
        ########
        # No more than `um_cpu` processes for each `Pool`
        for _ in range(self.n_estimators // num_cpu):
            pool = Pool(num_cpu)
            results = pool.starmap(grow_func, [(X, y,
                self.m_collaborative_trees,
                self.min_samples_split,
                self.min_samples_leaf,
                self.alpha, self.max_depth,
                group_list,
                self.random_update) for i in range(num_cpu)])
            self.forests.extend(results)
            pool.close()
            pool.join()
        if self.n_estimators % num_cpu > 0:
            pool = Pool(self.n_estimators % num_cpu)
            results = pool.starmap(grow_func, [(X, y,
                self.m_collaborative_trees,
                self.min_samples_split,
                self.min_samples_leaf,
                self.alpha, self.max_depth,
                group_list,
                self.random_update) for i in \
                            range(self.n_estimators % num_cpu)])
            self.forests.extend(results)
            pool.close()
            pool.join()

        # Calculate feature importance measures.
        self.importance(group_list = group_list)


    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            group_list: list[list[int]] | None = None) -> None:
        '''
        

        Parameters
        ----------
        X : np.ndarray

        y : np.ndarray

        group_list : list[list[int]] | None, optional
            A list of lists of indices of feature groups. 
            The default is None.

        Returns
        -------
        None
            Fit a Collaborative Trees Ensemble model without 
            multiprocessing. Calculate the feature importance.

        '''
        self.y_mean = np.mean(y, axis = 0)
        y = y - np.mean(y, axis = 0)
        self.y = y
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)


        # If self.n_bins is not None, then all solo features are
        # transformed into groups of one-hot indicators
        if self.n_bins is not None:
            self.X = X
            if group_list is None:
                # Indices of all continuous variables that will be
                # transformed into groups
                self.ind_var = [i for i in range(X.shape[1])]
                self.group_list_ori = \
                    [[i] for i in range(X.shape[1])]
            else:
                # Extract indices of all continuous variables                
                self.ind_var = [g[0] for g in group_list 
                                if len(g) == 1]
                self.group_list_ori = group_list
            X, group_list = data_transform(X,
                                           None,
                                           self.n_bins,
                                           self.ind_var,
                                           group_list)
        self.group_list = group_list
        self.n_features = X.shape[1]
        self.k_class = y.shape[1]
        self.forests = []
        
        
        #######
        if not group_list is None:
            checker = isinstance(group_list, list)
            if checker:
                for elem in group_list:
                    if not isinstance(elem, list):
                        checker = False
            if checker:
                list_temp = [y for elem in group_list for y in elem]
                if list_temp != [l for l in range(self.n_features)]:
                    return ('group_list should consist of column'
                           'indices that are collectively a'
                              'partition of all column indices')
            else:
                return ("group_list should be list of lists, where"
                        "each inner list consists of column indices"
                        "and all inner lists colletively are a"
                        "partition of column indices")
        if not group_list is None:
            for group in [x for x in group_list if len(x) > 1]:
                total_ = np.sum(X[:, group] == 0) \
                    + np.sum(X[:, group] == 1)
                for i in range(X.shape[0]):
                    if np.sum(X[i, group]) != 1 \
                        or total_ != X.shape[0] * len(group):
                        return('Variables in your groups may not be\
                               one-hot indicators')
        #######
        for i in range(self.n_estimators):
            forest = CollaborativeTreesClassifier(
                m_collaborative_trees = self.m_collaborative_trees,
                min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                alpha = self.alpha,
                max_depth = self.max_depth,
                random_update = self.random_update,
                group_list = group_list)
            forest.fit(X, y)
            self.forests.append(forest)

        #######
        # Calculate feature importance measures
        # If there are too many features, we should focus on only 
        # some of interesting features to save memory.
        self.importance(group_list = group_list)

    def importance(self, group_list = None):
        y = self.y
        if group_list is None:
            B = self.n_features
        else:
            B = len(group_list)

        self.importance_matrix = np.zeros(B**2)\
            .reshape(B, B)
        self.importance_addi = np.zeros(B)
        self.MDI = np.zeros(B)

        for forest in self.forests:
            forest.importance(group_list = group_list)
            self.importance_matrix = self.importance_matrix\
                + forest.importance_matrix
            self.importance_addi = self.importance_addi\
                + forest.importance_addi
            self.MDI = self.MDI\
                + forest.mdi_total

        self.importance_matrix = self.importance_matrix \
            / self.n_estimators
        self.importance_addi = self.importance_addi \
            / self.n_estimators
        self.MDI = self.MDI \
            / self.n_estimators

        ######
        # `np.sum(self.importance_matrix, axis=1)` + 
        # `self.importance_addi` is equivalent to `self.MDI`
        #######

        # Extended MDI:
        matrix_temp = self.importance_matrix + \
               self.importance_matrix.T
        self.exMDI = self.importance_addi + \
            np.sum(matrix_temp, axis = 1)


        self.diagram_pack_ori = []
        self.diagram_pack_ori.append(self.exMDI)
        self.diagram_pack_ori.append(self.importance_addi)
        self.diagram_pack_ori.append(matrix_temp)
        self.diagram_pack_ori.append('diagram_pack_ori[0] is the' 
             'XMDI vector. diagram_pack_ori[1] is'
            'the XMDI_ii vector (additive effects).'
                'diagram_pack[2] is the interaction'
            'matrix based on XMDI_ij, the diagonal entries are zero')


        self.diagram_pack = []
        self.diagram_pack.append(self.exMDI / np.var(y))
        denominator = np.copy(self.exMDI)                    
        denominator[denominator == 0] = float('Inf')
        self.diagram_pack.append(self.importance_addi / denominator)
        self.diagram_pack.append(matrix_temp)
        self.diagram_pack.append(matrix_temp / denominator)
        self.diagram_pack.append('diagram_pack[0] is'
            'XMDI_{i} / Var(Y). diagram_pack[1] is XMDI_{ii}'
            'standardized by XMDI_{i}. diagram_pack[2] is XMDI_{ij}'
            'matrix with zero diagonal entries. diagram_pack[3] is'
            'diagram_pack[2] standardized by XMDI_{i}')


        ######        
        # self.MDI : numpy array
        #   Usual MDI for trees
        # self.exMDI : numpy array 
        #   Extended MDI for collaborative trees
        # self.importance_addi : numpy array
        #   The measure of additive effect from each feature
        # self.importance_matrix : numpy matrix
        #   Two-ways interaction from each pair of features
        # self.diagram_pack_ori : a list
        #   self.diagram_pack[0] is XMDI_{i}'s
        #   self.diagram_pack[1] is XMDI_{ii}'s
        #   self.diagram_pack[2] is XMDI_{ij}'s (XMDI_{ii}'s are zero)
        #   self.diagram_pack[3] is the color's intensity in each 
        # self.diagram_pack : a list
        #   self.diagram_pack[0] is the standardized XMDI_{i}'s
        #   self.diagram_pack[1] is the standardized XMDI_{ii}'s
        #   self.diagram_pack[2] is XMDI_{ij}'s (XMDI_{ii}'s are zero)
        #   self.diagram_pack[3] is XMDI_{ij}'s / XMDI_{ii}'s
        ######
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        '''
        

        Parameters
        ----------
        X_test : np.ndarray

        Returns
        -------
        np.ndarray
            Predicted values at X_test.

        '''
        if self.n_bins is not None:
            # `group_list` is not needed here for making prediction.
            # `self.ind_var` is the list of indices of all continuous
            # variables that are transformed into groups
            X_test, group_list = data_transform(
                self.X,
                X_test,
                self.n_bins,
                self.ind_var,
                self.group_list_ori
                )
        
        
        predictions = np.zeros(X_test.shape[0])
        for i in range(self.n_estimators):
            forest = self.forests[i]
            predictions = predictions + forest.predict(X_test)
        predictions = predictions / self.n_estimators
        predictions = predictions[:, 0]
        return predictions + self.y_mean

def data_transform(
        X: np.ndarray,
        X_test: np.ndarray | None,
        n_bins: int,
        ind_var: list | None,
        group_list: list[list[int]] | None
        ) -> tuple[np.ndarray, list[list[int]]]:
    '''
    

    Parameters
    ----------
    X : np.ndarray
        An input n by p feature matrix.
    X_test : np.ndarray | None, optional
        The matrix to be binned. By default, if set to None, 
        X will be assigned to X_test.
    n_bins : int
        THe number of bins used.
    ind_var : list | None
        Indices of continuous features.
    group_list : list[list[int]] | None
        A list of lists of indices of feature groups.

    Returns
    -------
    X_new : np.ndarray
        A binned feature matrix.
    group_list_new : list[list[int]]
        A new feature group list.

    '''
    X_new, group_list_new = x_to_group_of_indicators(
        X,
        X_test = X_test,
        n_bins = n_bins,
        ind_var = ind_var,
        group_list = group_list)
    return X_new, group_list_new

def grow_func(
        X: np.ndarray,
        y: np.ndarray,
        m_collaborative_trees: int,
        min_samples_split: int,
        min_samples_leaf: int,
        alpha: float,
        max_depth: float,
        group_list: list[list[int]] | None,
        random_update: float
        ) -> CollaborativeTreesClassifier:
    '''
    

    Parameters
    ----------
    X : np.adarray

    y : np.adarray

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
    group_list : list[list[int]]| None, optional
        A list of lists of indices of feature groups. 
        The default is None.

    Returns
    -------
    CollaborativeTreesClassifier
        Fit and return a Collaborative Trees model.

    '''
    forest = CollaborativeTreesClassifier(
        m_collaborative_trees = m_collaborative_trees, 
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        random_update = random_update, 
        alpha = alpha,
        max_depth = max_depth, 
        group_list = group_list)

    forest.fit(X, y)

    return forest
