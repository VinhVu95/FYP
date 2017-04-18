from sklearn.base import BaseEstimator


class Regressor(object):

    def __init__(self, name, estimator, tuned_parameters):
        self.name = name
        self.estimator = estimator
        self.tuned_parameter = tuned_parameters
        self.is_regressor = True
        if isinstance(self.estimator, BaseEstimator):
            raise RuntimeError("Please use sklearn estimators")
        if isinstance(self.tuned_parameter, dict):
            raise RuntimeError("Please use Python dict to store parameters space")

    def __repr__(self):
        return self.name + " regression"

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self.estimator


class Classifier(object):

    def __init__(self, name, estimator, tuned_parameters):
        self.name = name
        self.estimator = estimator
        self.tuned_parameter = tuned_parameters
        self.is_classifier = True
        if isinstance(self.estimator, BaseEstimator):
            raise RuntimeError("Please use sklearn estimators")
        if isinstance(self.tuned_parameter, dict):
            raise RuntimeError("Please use Python dict to store parameters space")

    def __repr__(self):
        return self.name + " classification"

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self.estimator




