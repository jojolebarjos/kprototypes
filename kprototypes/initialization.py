
import numpy as np


def random_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_similarity,
    categorical_similarity,
    random_state,
):
    """Random initialization.

    Choose random points as cluster centroids.

    """

    n_points, _ = numerical_values.shape
    assert n_points >= n_clusters

    indices = random_state.permutation(n_points)
    selected_indices = indices[:n_clusters]
    # TODO maybe sample numerical centroids from normal distribution instead?
    numerical_centroids = numerical_values[selected_indices]
    categorical_centroids = categorical_values[selected_indices]
    return numerical_centroids, categorical_centroids


def huang_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_similarity,
    categorical_similarity,
    random_state,
):
    """Huang initialization.

    Based on "Clustering large data sets with mixed numeric and categorical
    values" by Huang (1997).

    """

    raise NotImplementedError()


def cao_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_similarity,
    categorical_similarity,
    random_state,
):
    """Cao et al. initialization.

    Based on "A new initialization method for categorical data clustering" by
    Cao et al. (2009).

    """

    n_points, n_categorical_features = categorical_values.shape

    # Numerical centroids are sampled from normal distribution
    numerical_centroids = _initialize_numerical_from_normal(
        numerical_values,
        n_clusters,
        random_state,
    )

    # Categorical representative are sampled from training set
    categorical_centroids = np.empty((n_clusters, n_categorical_features), dtype=np.int32)

    # Compute pointwise density
    density = np.zeros(n_points, dtype=np.int32)
    for j in range(n_categorical_features):
        categorical_value = categorical_values[:, j]
        frequency = np.bincount(categorical_value)
        density += frequency[categorical_value]
    density = density.astype(np.float32) / (n_points * n_categorical_features)

    # First cluster is most frequent point
    point = np.argmax(density)
    categorical_centroids[0] = categorical_values[point]

    # Then, choose the most dissimilar point at each step, with respect to current cluster set
    for k in range(1, n_clusters):
        costs = categorical_similarity(
            categorical_values[:, None],
            categorical_centroids[None, :k]
        )
        weighted_costs = costs * density[:, None]
        min_weighted_costs = weighted_costs.min(axis=1)
        point = categorical_values[np.argmax(min_weighted_costs)]
        categorical_centroids[k] = point

    return categorical_centroids


def _initialize_numerical_from_normal(numerical_values, n_clusters, random_state):
    """Sample numerical centroids from normal distribution."""

    _, n_numerical_features = numerical_values.shape
    mean = np.mean(numerical_values, axis=0)
    std = np.std(numerical_values, axis=0)
    numerical_centroids = random_state.randn(n_clusters, n_numerical_features) * std + mean
    return numerical_centroids
