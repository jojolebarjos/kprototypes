
import numpy as np
import pandas as pd

from .optimization import do_iteration


def train(X, **kwargs):
    raise NotImplementedError()


def train_dataframe(df, **kwargs):
    raise NotImplementedError()


def train_arrays(
    numerical_values,
    categorical_values,
    *,
    n_clusters,
    centroid_initialization,
    numerical_similarity,
    categorical_similarity,
    gamma,
    n_iterations,
    random_state,
    verbose,
):

    # TODO default values?
    # TODO verbosity

    n_points, n_numerical_features = numerical_values.shape
    _, n_categorical_features = categorical_values.shape

    # Initialize centroids
    numerical_centroids, categorical_centroids = centroid_initialization(
        numerical_values,
        categorical_values,
        n_clusters,
        numerical_similarity,
        categorical_similarity,
        random_state,
    )

    # Assign points to clusters
    # TODO paper does update clusters after each assignment
    clustership = ...

    # Keep track of point count per cluster
    cluster_counts = np.bincount(clustership, minlength=n_clusters)

    # Keep track of numerical features sums per cluster
    one_hot = np.eye(n_clusters, dtype=np.bool)[clustership]
    cluster_sums = numerical_values[one_hot].sum(axis=0)

    # Enumerate categorical values
    categorical_counts = categorical_values.max(axis=0) + 1
    categorical_offsets = np.concatenate([[0], categorical_counts.cumsum()])
    n_categorical_values = categorical_offsets[-1]

    # Keep track of categorical values counts per cluster
    # Note: this implementation packs all possible values in a flat array
    # TODO optimize this (possibly just jit)
    cluster_frequencies = np.zeros((n_clusters, n_categorical_values), dtype=np.int32)
    for i in range(n_points):
        for j in range(n_categorical_features):
            offset = categorical_offsets[j] + categorical_values[i, j]
            cluster_frequencies[k, offset] += 1

    # Run for several epochs
    for iteration in range(n_iterations):
        moves = do_iteration(
            numerical_values,
            categorical_values,
            numerical_centroids,
            categorical_centroids,
            numerical_similarity,
            categorical_similarity,
            gamma,
            categorical_offsets,
            clustership,
            cluster_counts,
            cluster_sums,
            cluster_frequencies,
        )

    return (
        numerical_centroids,
        categorical_centroids,
        clustership,
        # TODO moves? iterations? costs?
    )
