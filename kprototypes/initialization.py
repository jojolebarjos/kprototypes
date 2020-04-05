
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

    Used in "Clustering large data sets with mixed numeric and categorical
    values" by Huang (1997), the original k-prototypes definition.

    """

    n_points, _ = numerical_values.shape
    assert n_points >= n_clusters

    indices = random_state.permutation(n_points)
    selected_indices = indices[:n_clusters]
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
    verbose,
):
    """Frequency-based initialization.

    Select points that are the most representative of frequent features.

    Based on "Extensions to the k-modes algorithm for clustering large data
    sets with categorical values" by Huang (1998).

    """

    raise NotImplementedError()


def cao_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_similarity,
    categorical_similarity,
    random_state,
    verbose,
):
    """Cao et al. initialization.

    Based on "A new initialization method for categorical data clustering" by
    Cao et al. (2009).

    """

    n_points, n_numerical_features = numerical_values.shape
    _, n_categorical_features = categorical_values.shape
    assert n_points >= n_clusters

    # Allocate centroid arrays
    numerical_centroids = np.empty(
        (n_clusters, n_numerical_features),
        dtype=numerical_values.dtype,
    )
    categorical_centroids = np.empty(
        (n_clusters, n_categorical_features),
        dtype=categorical_values.dtype,
    )

    # Compute pointwise density for categorical values
    # TODO Should take into account numerical features as well...
    #      Maybe gamma-weighted sum, using distance to mean for numericals?
    density = np.zeros(n_points, dtype=np.int32)
    for j in range(n_categorical_features):
        categorical_value = categorical_values[:, j]
        frequency = np.bincount(categorical_value)
        density += frequency[categorical_value]
    density = density.astype(np.float32) / (n_points * n_categorical_features)

    # First cluster is most frequent point
    point = np.argmax(density)
    numerical_centroids[0] = numerical_values[point]
    categorical_centroids[0] = categorical_values[point]

    # Then, choose the most dissimilar point at each step, with respect to current cluster set
    for k in range(1, n_clusters):
        raise NotImplementedError()
        costs = categorical_similarity(
            categorical_values[:, None],
            categorical_centroids[None, :k]
        )
        weighted_costs = costs * density[:, None]
        min_weighted_costs = weighted_costs.min(axis=1)
        point = categorical_values[np.argmax(min_weighted_costs)]
        categorical_centroids[k] = point

    return categorical_centroids
