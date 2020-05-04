
import numpy as np

from sklearn.neighbors import KernelDensity


def check_initialization(initialization):
    if initialization is None:
        return random_initialization
    if callable(initialization):
        return initialization
    if initialization == 'random':
        return random_initialization
    if initialization == 'frequency':
        return frequency_initialization
    if isinstance(initialization, (tuple, list)):
        assert len(initialization) == 2
        return _explicit_initialization_factory(*initialization)
    raise KeyError(initialization)


def _explicit_initialization_factory(numerical_centroids, categorical_centroids):
    """Create dummy initialization method, returning precomputed centroids."""

    # TODO check dtype and shape

    def initialization(
        numerical_values,
        categorical_values,
        n_clusters,
        numerical_distance,
        categorical_distance,
        gamma,
        random_state,
        verbose,
    ):
        assert numerical_centroids.shape[1] == n_clusters
        assert categorical_centroids.shape[1] == n_clusters
        return numerical_centroids, categorical_centroids

    return initialization


def random_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_distance,
    categorical_distance,
    gamma,
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

    # TODO need to discard duplicates?
    indices = random_state.permutation(n_points)
    selected_indices = indices[:n_clusters]
    numerical_centroids = numerical_values[selected_indices]
    categorical_centroids = categorical_values[selected_indices]
    return numerical_centroids, categorical_centroids


# TODO "Extensions to the k-modes algorithm for clustering large data sets with categorical values" by Huang (1998)?


def _numerical_density(values):
    """Estimate density of a continous random variable."""

    n_points, n_features = values.shape
    densities = np.zeros((n_points, n_features), dtype=np.float32)
    for j in range(n_features):
        v = values[:, j, None]
        kde = KernelDensity()
        kde.fit(v)
        log_densities = kde.score_samples(v)
        densities[:, j] = np.exp(log_densities)
    return densities


def _categorical_density(values):
    """Estimate density of a discrete random variable."""

    n_points, n_features = values.shape
    densities = np.zeros((n_points, n_features), dtype=np.int32)
    for j in range(n_features):
        frequencies = np.bincount(values[:, j])
        densities[:, j] = frequencies[values[:, j]]
    densities = densities.astype(np.float32)
    densities /= n_points
    return densities


def frequency_initialization(
    numerical_values,
    categorical_values,
    n_clusters,
    numerical_distance,
    categorical_distance,
    gamma,
    random_state,
    verbose,
):
    """Frequency-based initialization.

    ...

    This is an extension for mixed values of "A new initialization method for
    categorical data clustering" by Cao et al. (2009).

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

    # Estimate probability of each sample and each feature
    # TODO should maybe stick to log-space for stability?
    densities = np.concatenate([
        _numerical_density(numerical_values),
        _categorical_density(categorical_values)
    ], axis=1)

    # Mean density is used as weight
    weights = densities.mean(axis=1)

    # First cluster is most likely point
    index = np.argmax(weights)
    numerical_centroids[0] = numerical_values[index]
    categorical_centroids[0] = categorical_values[index]

    # Then, choose the most dissimilar point at each step, with respect to current cluster set
    for k in range(1, n_clusters):

        # Compute distance w.r.t. already initialized centroids
        numerical_costs = numerical_distance(
            numerical_values[:, None],
            numerical_centroids[None, :k]
        )
        categorical_costs = categorical_distance(
            categorical_values[:, None],
            categorical_centroids[None, :k]
        )
        costs = numerical_costs + gamma * categorical_costs

        # Maximize minimum distance (i.e. ensure largest margin)
        weighted_costs = costs * weights[:, None]
        min_weighted_costs = weighted_costs.min(axis=1)
        index = np.argmax(min_weighted_costs)
        numerical_centroids[k] = numerical_values[index]
        categorical_centroids[k] = categorical_values[index]

    return numerical_centroids, categorical_centroids
