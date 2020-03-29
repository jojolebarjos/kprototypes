
import numpy as np

from numba import jit


@jit(nopython=True)
def do_iteration(
    numerical_values,
    categorical_values,
    numerical_centroids,
    categorical_centroids,
    numerical_similarity,
    categorical_similarity,
    gamma,
    categorical_offsets,
    clustership,
    cluster_count,
    cluster_sum,
    cluster_frequency,
):

    n_points, n_categorical_features = categorical_values.shape

    moves = 0
    for i in range(n_points):

        numerical_value = numerical_values[i]
        categorical_value = categorical_values[i]

        # TODO probably need to pass in-place? or just don't use vectorized stuff
        numerical_cost = numerical_similarity(numerical_value, numerical_centroids)
        categorical_cost = categorical_similarity(numerical_value, categorical_centroids)
        cost = numerical_cost + gamma * categorical_cost

        old_cluster = clustership[i]
        new_cluster = cost.argmin()

        if old_cluster != new_cluster:

            # Update point count for both clusters
            cluster_count[old_cluster] -= 1
            cluster_count[new_cluster] += 1

            # Update numerical features sums for both clusters
            cluster_sum[old_cluster] -= numerical_value
            cluster_sum[new_cluster] += numerical_value

            # Update categorical values counts for both clusters
            for j in range(n_categorical_features):
                offset = categorical_offsets[j] + categorical_value[j]
                cluster_frequency[old_cluster, offset] -= 1
                cluster_frequency[new_cluster, offset] += 1

            moves += 1

    return moves
