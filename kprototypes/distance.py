
import numpy as np


def check_distance(distance):
    if distance is None:
        return euclidean_distance
    if callable(distance):
        return distance
    if distance == 'euclidean':
        return euclidean_distance
    if distance == 'manhattan':
        return manhattan_distance
    if distance == 'matching':
        return matching_distance
    raise KeyError(distance)


def euclidean_distance(a, b):
    """Euclidean distance."""

    return np.sum((a - b) ** 2, axis=-1)


def manhattan_distance(a, b):
    """Manhattan distance."""

    return np.abs(a - b).sum(axis=-1)


def matching_distance(a, b):
    """Matching distance."""

    return np.sum(a != b, axis=-1)


# TODO Jaccard distance
