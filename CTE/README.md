# Collaborative Trees Ensemble


Collaborative Trees Ensemble is a tree-based model designed for comprehensive feature analysis. It is capable of inferring overall feature importance, additive effects, and interaction effects. The package provides functions for generating network diagrams to visualize these effects, offering insights into complex relationships within your data.

With Collaborative Trees Ensemble, you can efficiently analyze feature interactions and uncover nuanced patterns in your datasets. Whether you're working on predictive modeling, exploratory data analysis, or feature engineering tasks, this package provides powerful tools to enhance your workflow.


## Installation

Before installing Collaborative Trees Ensemble, ensure you have the following dependencies installed:

* Spyder version: 5.5.1 (conda)
* Python version: 3.10.12 64-bit

You can install Collaborative Trees Ensemble and its dependencies using the following command:

```conda install xgboost==1.5.0 sklearn scipy hyperopt matplotlib networkx openml```


## Dependencies

xgboost: Used for boosting algorithms (version 1.5.0).
sklearn: Provides machine learning algorithms and utilities (version 1.2.2).
scipy: Scientific computing library for numerical operations (version 1.11.1).
hyperopt: Library for hyperparameter optimization (version 0.2.7).
matplotlib: Plotting library for visualization (version 3.7.2).
networkx: Library for graph-based algorithms (version 3.1).
openml: Platform for sharing and exploring datasets (version 0.12.2).

## Usage

The following file demonstrate the basic usage of Collaborative Trees Ensemble.

>main_py/example.py

The following file is for the embryo dataset anaylsis in the paper.

>main_py/embryogrowth.py

The following files are for the openML dataset anaylsis in the paper.

>main_py/open_ml.py
>
>main_py/openml_score_print.py
>
>main_py/open_ml_download_datasets.py

The following files are for the simulation experiments in Section 5 of the paper.

>main_py/example_bias.py
>
>main_py/example_plot.py



## Example
```
import numpy as np

from method.cte import CollaborativeTreesEnsemble 
from method.util.param_search import search_start
from method.util.plot_network import plot_network_start

X = np.random.multivariate_normal(
    [0], [[1]], size = n * p).reshape(n, p)
y = 5 * X[:, 0] + 10 * X[:, 1]
y = y + np.squeeze(np.random.multivariate_normal(
                        [0], [[1]], size = y.shape[0]))

best_param = {'n_trees': 9,
     'min_samples_split': 15,
     'min_samples_leaf': 20,
     'random_update': 0.0001,
     'alpha': 10000.0,
     'max_depth': 10,
     'n_bins': None}
forest = CollaborativeTreesEnsemble(n_estimators = 100,
        dict_param = best_param)

forest.fit(X, y)

# Call the function to generate the network plot:
parameters = {'base_size': 8500, 'base_edge_size': 4,
              'horizontal_positive_shift': 0.1,
              'horizontal_negative_shift': 0.0,
              'vertical_positive_shift': 0.15,
              'vertical_negative_shift': -0.1,
              'label_font_size': 40, 'edge_label_font_size': 25}
plot_network_start(forest.diagram_pack, parameters, digits = 1)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information

Chien-Ming CHI

Institute of Statistical Science

Academia Sinica

xbb66kw at gmail.com


