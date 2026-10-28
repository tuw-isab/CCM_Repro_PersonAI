import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mord import LogisticIT, LogisticAT, OrdinalRidge

from personai_pcms.constants import RANDOM_SEED, LABELS
from personai_pcms.utils.utils import get_pipeline_params


def make_random_forest_classifier():
    clf = RandomForestClassifier(random_state=RANDOM_SEED)
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params


def make_support_vector_classifier():
    clf = SVC(random_state=RANDOM_SEED)
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params


def make_logistic_regression_classifier():
    clf = LogisticRegression(random_state=RANDOM_SEED)
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params


def make_random_forest_classifier_with_standard_scaler():
    clf = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            ("random_forest_classifier", RandomForestClassifier(random_state=RANDOM_SEED)),
        ]
    )
    params = get_pipeline_params(clf)
    model_class = clf.__class__.__name__
    return clf, model_class, params


def make_logistic_regression_classifier_with_standard_scaler():
    clf = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            ("logistic_regression_classifier", LogisticRegression(random_state=RANDOM_SEED)),
        ]
    )
    params = get_pipeline_params(clf)
    model_class = clf.__class__.__name__
    return clf, model_class, params

class MajorityClassParrot(ClassifierMixin, BaseEstimator):
    def __init__(self, *, pred_class=None):
        self.majority_class = pred_class

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(shape=X.shape[0], fill_value=self.majority_class)

    def get_params(self, deep=True):
        return {'majority_class': self.majority_class}


class DistributionParrot(ClassifierMixin, BaseEstimator):
    def __init__(self, *, probs=None, vals=None):
        self.probs = probs
        self.vals = vals

    def fit(self, X, y):
        vals, counts = np.unique(np.random.choice(y,10), return_counts=True)
        self.vals = vals
        self.probs = counts / counts.sum()
        return self

    def predict(self, X):
        return np.random.choice(self.vals, size=X.shape[0], p=self.probs)

    def get_params(self, deep=True):
        return {'probs': self.probs, 'vals': self.vals}

class RandomClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.random.choice(LABELS, size=X.shape[0])

    def get_params(self, deep=True):
        return {}

def make_random_classifier():
    clf = RandomClassifier()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params

def make_distribution_parrot_classifier():
    clf = DistributionParrot()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params

def make_majority_class_parrot_classifier():
    clf = MajorityClassParrot()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params

def make_ordinal_logistic_regression_classifier_with_mord_IT():
    clf = LogisticIT()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params

def make_ordinal_logistic_regression_classifier_with_mord_AT():
    clf = LogisticAT()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params

def make_ordinal_ridge_classifier_with_mord():
    clf = OrdinalRidge()
    params = clf.get_params()
    model_class = clf.__class__.__name__
    return clf, model_class, params
