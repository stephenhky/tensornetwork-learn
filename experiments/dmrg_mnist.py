
import numpy as np
import tensornetwork as tn
from mlexpt.ml.core import ExperimentalClassifier


class TNDigitsClassifier(ExperimentalClassifier):
    def fit(self, X, Y, *args, **kwargs):
        pass

    def fit_batch(self, dataset, *args, **kwargs):
        pass

    def predict_proba(self, X, *args, **kwargs):
        pass

    def predict_proba_batch(self, dataset, *args, **kwargs):
        pass

    def persist(self, path):
        pass

    def trim(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        pass