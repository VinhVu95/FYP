from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVR
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

import subprocess
import numpy as np


def regression_families():
    candidates = []

    # Support vector regressor
    svr = {'name': 'support vector regression', 'estimator': SVR()}
    svr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1.10, 100, 1000, 10000], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    candidates.append([svr['name'], svr, svr_tuned_parameters])

    # Random Forest Regressor
    rfr = {'name': 'Random Forest Regressor',
           'estimator': RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=50, min_samples_split=8, min_samples_leaf=2)}
    rfr_tuned_parameters = {'n_estimators': [300, 500, 800, 1200],
                            # "criterion": ["gini", "entropy"],
                            # 'min_samples_split': np.arange(2, 10),
                            # 'min_samples_leaf': np.arange(2, 10),
                            #'oob_score': [True, False],
                            'max_features': ['log2', 'sqrt', None]}
    candidates.append([rfr['name'], rfr, rfr_tuned_parameters])

    # Linear Regression
    lnr = {'name': 'Linear Regression', 'estimator': LinearRegression()}
    lnr_tuned_parameters = {'fit_intercept': [True, False],
                            'normalize': [True, False]}
    candidates.append([lnr['name'], lnr, lnr_tuned_parameters])

    # Ridge regressor
    rir = {'name': 'Ridge regression', 'estimator': Ridge()}
    rir_tuned_parameters = {'alpha': [0.01, 0.1, 1.0, 10, 100],
                            'fit_intercept': [True, False], 'normalize': [True, False]}
    candidates.append([rir['name'], rir, rir_tuned_parameters])

    # # Extra tree regressor
    # etr = {'name': 'Extra tree regression', 'estimator': ExtraTreesRegressor(n_jobs=-1)}
    # etr_tuned_parameters = {'n_estimators': [300, 500, 800],
    #                         'max_features': ['log2', 'sqrt', None]}
    # candidates.append([etr['name'], etr, etr_tuned_parameters])

    # Lasso
    lasso = {'name': 'Lasso', 'estimator': Lasso()}
    lasso_tuned_parameters = {'alpha': [0.1, 1.0, 10, 100],
                              'normalize': [True, False]}
    candidates.append([lasso['name'], lasso, lasso_tuned_parameters])

    # Neural network
    nn = {'name': 'Neural Network Regressor', 'estimator': MLPRegressor()}
    nn_tuned_parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                           'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100]}
    candidates.append([nn['name'], nn, nn_tuned_parameters])

    # Elastic Net
    # enet = {'name': 'Elastic Net', 'estimator': ElasticNet(l1_ratio=0)}
    # enet_tuned_parameters = {'alpha': [0.1, 1.0, 10, 100],
    #                          'normalize': [True, False]}
    # candidates.append([enet['name'], enet, enet_tuned_parameters])

    # Decision tree regression
    # dt = {'name': 'Decision Tree Regression', 'estimator': DecisionTreeRegressor()}
    # dt_tuned_parameters = {'max_depth': [2, 5, 10, 20]}
    # candidates.append([dt['name'], dt, dt_tuned_parameters])

    # Logistic Regression
    # lr = {'name': 'Logistic Regression', 'estimator': LogisticRegression()}
    # lr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    # candidates.append([lr['name'], lr, lr_tuned_parameters])

    # K nearest neighbor regression
    knn = {'name': 'K Nearest Neighbors', 'estimator': KNeighborsRegressor()}
    knn_tuned_parameters = {'weights': ['uniform', 'distance']}
    candidates.append([knn['name'], knn, knn_tuned_parameters])

    return candidates


def classification_families():
    candidates = []

    # Decision Tree Classifier
    dtc = {'name': 'Decision Tree Classification', 'estimator': DecisionTreeClassifier()}
    dtc_tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20]
              # "max_depth": [None, 2, 5, 10],
              # "min_samples_leaf": [1, 5, 10],
              # "max_leaf_nodes": [None, 5, 10, 20],
              }
    candidates.append([dtc['name'], dtc, dtc_tuned_parameters])

    # Random Forest Classifier
    rfc = {'name': 'Random Forest Classification', 'estimator': RandomForestClassifier()}
    rfc_tuned_parameters = {'n_estimators': [50, 300, 500, 800],
                            # "criterion": ["gini", "entropy"],
                            # 'min_samples_split': [2, 10, 20],
                            # 'min_samples_leaf': [1, 2, 5, 10],
                            'max_features': ['log2', 'sqrt', None]}
    candidates.append([rfc['name'], rfc, rfc_tuned_parameters])

    # Logistic Regression
    lr = {'name': 'Logistic Regression', 'estimator': LogisticRegression(multi_class='ovr')}
    lr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    candidates.append([lr['name'], lr, lr_tuned_parameters])

    # Naive Bayes
    nbc = {'name': 'Gaussian Naive Bayes', 'estimator': GaussianNB()}
    nbc_tuned_parameters = {}
    candidates.append([nbc['name'], nbc, nbc_tuned_parameters])

    return candidates


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")