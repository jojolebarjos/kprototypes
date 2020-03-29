
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


# TODO see https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/cluster/_kmeans.py


class KPrototypes(ClusterMixin, BaseEstimator):
    """K-Prototypes clustering.
    
    ...
    
    """

    def __init__(self,
        n_clusters=8,
        random_state=None,
        # TODO
    ):

        self.n_clusters = n_clusters
        self.random_state = random_state
        ...

    def fit(self, X, y=None):
        ...
        return self

    def fit_predict(self, X, y=None):
        ...
