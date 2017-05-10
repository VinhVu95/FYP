import subprocess

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge,Lasso, LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from classes.predictor import Regressor,Classifier


def regression_families():
    candidates = []

    # Support vector regressor
    svr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1.10, 100, 1000, 10000],
                            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svr = Regressor('support vector regression',SVR(), svr_tuned_parameters)
    candidates.append(svr)

    # Linear Regression
    lnr_tuned_parameters = {'fit_intercept': [True, False],
                            'normalize': [True, False]}
    lnr = Regressor('Linear Regression',LinearRegression(),lnr_tuned_parameters)
    candidates.append(lnr)

    # Ridge regressor
    rir_tuned_parameters = {'alpha': [0.01, 0.1, 1.0, 10, 100],
                            'fit_intercept': [True, False], 'normalize': [True, False]}
    rir = Regressor('Ridge regression', Ridge(), rir_tuned_parameters)
    candidates.append(rir)

    # # Extra tree regressor
    # etr = {'name': 'Extra tree regression', 'estimator': ExtraTreesRegressor(n_jobs=-1)}
    # etr_tuned_parameters = {'n_estimators': [300, 500, 800],
    #                         'max_features': ['log2', 'sqrt', None]}
    # candidates.append([etr['name'], etr, etr_tuned_parameters])

    # Lasso
    lasso_tuned_parameters = {'alpha': [0.1, 1.0, 10, 100],
                              'normalize': [True, False]}
    lasso = Regressor('Lasso',Lasso(),lasso_tuned_parameters)
    candidates.append(lasso)

    # Neural network
    nn_tuned_parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                           'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100]}
    nn = Regressor('Neural Network Regressor',MLPRegressor(),nn_tuned_parameters)
    candidates.append(nn)

    # Random Forest Regressor
    rfr_tuned_parameters = {'n_estimators': [800],
                            # "criterion": ["gini", "entropy"],
                            # 'min_samples_split': np.arange(2, 10),
                            # 'min_samples_leaf': np.arange(2, 10),
                            # 'oob_score': [True, False],
                            'max_features': ['sqrt', None]}
    rfr = Regressor('Random Forest Regressor',RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=50, min_samples_split=8,
                                              min_samples_leaf=2), rfr_tuned_parameters)
    candidates.append(rfr)

    # Elastic Net
    # enet = {'name': 'Elastic Net', 'estimator': ElasticNet(l1_ratio=0)}
    # enet_tuned_parameters = {'alpha': [0.1, 1.0, 10, 100],
    #                          'normalize': [True, False]}
    # candidates.append([enet['name'], enet, enet_tuned_parameters])
    #
    # Decision tree regression
    # dt = {'name': 'Decision Tree Regression', 'estimator': DecisionTreeRegressor()}
    # dt_tuned_parameters = {'max_depth': [2, 5, 10, 20]}
    # candidates.append([dt['name'], dt, dt_tuned_parameters])
    #
    # Logistic Regression
    # lr = {'name': 'Logistic Regression', 'estimator': LogisticRegression()}
    # lr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    # candidates.append([lr['name'], lr, lr_tuned_parameters])

    # K nearest neighbor regression
    knn_tuned_parameters = {'weights': ['uniform', 'distance']}
    knn = Regressor('K Nearest Neighbors', KNeighborsRegressor(), knn_tuned_parameters)
    candidates.append(knn)

    return candidates


def classification_families():
    candidates = []

    # Decision Tree Classifier
    dtc_tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20]
              # "max_depth": [None, 2, 5, 10],
              # "min_samples_leaf": [1, 5, 10],
              # "max_leaf_nodes": [None, 5, 10, 20],
              }
    dtc = Classifier('Decision Tree Classification', DecisionTreeClassifier(),dtc_tuned_parameters)
    candidates.append(dtc)

    # Random Forest Classifier
    rfc_tuned_parameters = {'n_estimators': [50, 300, 500, 800],
                            # "criterion": ["gini", "entropy"],
                            # 'min_samples_split': [2, 10, 20],
                            # 'min_samples_leaf': [1, 2, 5, 10],
                            'max_features': ['log2', 'sqrt', None]}
    rfc = Classifier('Random Forest Classification',RandomForestClassifier(),rfc_tuned_parameters)
    candidates.append(rfc)

    # Logistic Regression
    lr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    lr = Classifier('Logistic Regression',LogisticRegression(multi_class='ovr'),lr_tuned_parameters)
    candidates.append(lr)

    # Naive Bayes
    nbc_tuned_parameters = {}
    nbc = Classifier('Gaussian Naive Bayes',GaussianNB(),nbc_tuned_parameters)
    candidates.append(nbc)

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
