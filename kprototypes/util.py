
from collections import Counter

import numpy as np


class CategoricalTransformer:

    UNKNOWN = object()

    def __init__(self, min_count=0, allow_unknown=True, nan_as_unknown=True):

        # Keep parameters
        if min_count > 0 and not allow_unknown:
            raise ValueError(
                "Cannot use min_count when unknown values are forbidden"
            )
        self.min_count = min_count
        self.allow_unknown = allow_unknown
        self.nan_as_unknown = nan_as_unknown

        # Parameters are not yet fitted
        self._table = None
        self._mapping = None

    def fit(self, values):

        # Count values, unifying NaN-likes
        counter = Counter()
        for value in values:
            if value != value:
                value = np.nan
            counter[value] += 1

        # Discard NaN, if considered as unknown value
        if self.nan_as_unknown and np.nan in counter:
            if not self.allow_unknown:
                raise ValueError(
                    "Cannot have NaN-like values as unknown when unknown "
                    "values are forbidden"
                )
            del counter[np.nan]

        # Create mapping table
        table = [v for v, c in counter.items() if c >= self.min_count]
        if self.allow_unknown:
            mapping = {v: i + 1 for i, v in enumerate(table)}
            table = [self.UNKNOWN, *table]
        else:
            mapping = {v: i for i, v in enumerate(table)}

        # Define mapping function
        if self.allow_unknown:
            def _map(value):
                if value != value:
                    value = np.nan
                return mapping.get(value, 0)
        else:
            def _map(value):
                if value != value:
                    value = np.nan
                return mapping[value]

        # Wrap as NumPy objects
        self._table = np.array(table, dtype=object)
        self._mapping = np.vectorize(_map, otypes=[np.int32])

        return self

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def transform(self, values):
        return self._mapping(values)

    def inverse_transform(self, indices):
        return self._table[indices]
