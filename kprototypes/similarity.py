
import numpy as np


def check_similarity(similarity):
    if similarity is None:
        return euclidean_similarity
    if callable(similarity):
        return similarity
    if similarity == 'euclidean':
        return euclidean_similarity
    if similarity == 'matching':
        return matching_similarity
    raise KeyError(similarity)


def euclidean_similarity(a, b):
    """Euclidean distance."""

    return np.sum((a - b) ** 2, axis=-1)


def matching_similarity(a, b):
    """Matching similarity."""

    return np.sum(a != b, axis=-1)


# TODO Jaccard similarity
