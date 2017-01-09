import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import learning_curve


# cross-validation score generator
# return list of scores for every fold
def cv_score_gen(model, input, output, nfolds):
    kfold = KFold(n_splits=nfolds, shuffle=True)
    return [model.fit(input.ix[train, :], output[train]).score(input.ix[test, :], output[test])
            for train, test in kfold.split(input)]

# TODO Feature selection


# TODO Grid search for hyper parameter tuning for a model
# return the best configuration for a model using cross validation and grid search
def best_config(model, parameters, train_instances, judgements):
    """
    Args:
        model: the model dict object with name of model and actual estimators
        parameters: dict of hyper parameter for the model
        train_instances: training data set
        judgements: training output data set

    Returns:  a triple representing the best configuration, the quality score (measure using accuracy) and the
    classifier object with the best configuration
    The parameters we have used in the GridSearch call are 5-fold cross-validation, with model selection based on
    accuracy, verbose output and 4 jobs running in parallel while tuning the parameters

    """
    print('Grid search for... ' + model['name'])
    print(model['estimator'])
    # print(train_instances)
    clf = GridSearchCV(estimator=model['estimator'], param_grid=parameters,
                       cv=5, verbose=4, scoring='r2')
    clf.fit(train_instances, judgements)
    best_estimator = clf.best_estimator_
    print('Best hyper parameters: ' + str(clf.best_params_))
    print('Best score of this configuration: ' + str(clf.best_score_))

    return [str(clf.best_params_), clf.best_score_,
            best_estimator]


# TODO Choosing estimators
# return best model from set of model families given training data using cross-validation
def best_model(classifier_families, train_instances, judgements):
    best_quality = 0.0
    best_classifier = None
    classifiers = []
    for name, model, parameters in classifier_families:
        classifiers.append(best_config(model, parameters,
                                       train_instances,
                                       judgements))

    for name, quality, classifier in classifiers:
        print('Considering classifier... ' + name + ' ' + str(quality))
        if quality > best_quality:
            best_quality = quality
            best_classifier = [name, classifier]

    print('Best classifier... ' + best_classifier[0] + ' best quality ' + str(best_quality))
    return best_classifier[1]


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt