
import numpy as np


def check_initialization(initialization):
    if initialization is None:
        return random_initialization
    if callable(initialization):
        return initialization
    if initialization == 'random':
        return random_initialization
    # TODO other initializations
    raise KeyError(initialization)


def random_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_similarity,
    categorical_similarity,
    random_state,
    verbose,
):
    """Random initialization.

    Choose random points as cluster centroids.

    """

    n_points, _ = numerical_values.shape
    assert n_points >= n_clusters

    indices = random_state.permutation(n_points)
    selected_indices = indices[:n_clusters]
    numerical_centroids = numerical_values[selected_indices]
    categorical_centroids = categorical_values[selected_indices]
    return numerical_centroids, categorical_centroids
