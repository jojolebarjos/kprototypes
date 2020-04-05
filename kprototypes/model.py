
import numpy as np

from sklearn.utils import check_array, check_random_state

from .similarity import check_similarity
from .initialization import check_initialization
from .optimization import fit, predict


class KPrototypes:
    """K-Prototypes clustering.

    ...

    """

    def __init__(self,
        n_clusters=8,
        initialization=None,
        numerical_similarity=None,
        categorical_similarity=None,
        gamma=None,
        n_iterations=100,
        random_state=None,
        verbose=0,
    ):

        # Resolve string-based properties
        self.initialization = check_initialization(initialization)
        self.numerical_similarity = check_similarity(numerical_similarity)
        self.categorical_similarity = check_similarity(categorical_similarity)

        # Gamma and random state will be resolved when fitted
        self.gamma = gamma
        self.random_state = random_state

        # Store other arguments, ensuring type
        self.n_clusters = int(n_clusters)
        self.n_iterations = int(n_iterations)
        self.verbose = bool(verbose)
        
        # Parameters are not yet fitted
        self.true_gamma = None
        self.numerical_centroids = None
        self.categorical_centroids = None

    def fit(self, numerical_values, categorical_values):

        # Regular fit, discarding cluster assignment
        self.fit_predict(numerical_values, categorical_values)
        return self

    def fit_predict(self, numerical_values, categorical_values):

        # Check input
        # TODO maybe ensure_min_features=0?
        numerical_values = check_array(
            numerical_values,
            dtype=[np.float32, np.float64],
        )
        categorical_values = check_array(
            categorical_values,
            dtype=[np.int32, np.int64],
        )

        # Estimate gamma, if not specified
        if self.gamma is None:
            gamma = 0.5 * numerical_values.std()
        else:
            gamma = float(self.gamma)

        # Resolve random state
        random_state = check_random_state(self.random_state)

        # Initialize clusters
        numerical_centroids, categorical_centroids = self.initialization(
            numerical_values,
            categorical_values,
            self.n_clusters,
            self.numerical_similarity,
            self.categorical_similarity,
            random_state,
            self.verbose,
        )

        # Train clusters
        clustership = fit(
            numerical_values,
            categorical_values,
            numerical_centroids,
            categorical_centroids,
            self.numerical_similarity,
            self.categorical_similarity,
            gamma,
            self.n_iterations,
            random_state,
            self.verbose,
        )

        # Save result
        self.true_gamma = gamma
        self.numerical_centroids = numerical_centroids
        self.categorical_centroids = categorical_centroids

        return clustership

    def predict(self, numerical_values, categorical_values):

        return predict(
            numerical_values,
            categorical_values,
            self.numerical_centroids,
            self.categorical_centroids,
            self.numerical_similarity,
            self.categorical_similarity,
            self.true_gamma,
        )
