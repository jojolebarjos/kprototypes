
import numpy as np

from numba import jit


@jit(nopython=True)
def matching_similarity(a, b):
    """Matching similarity."""

    return np.sum(a != b, axis=-1)


@jit(nopython=True)
def euclidean_similarity(a, b):
    """Euclidean distance."""

    return np.sum((a - b) ** 2, axis=-1)


# TODO Jaccard similarity
