
# K-prototypes

K-prototypes, as introduced by [Huang (1997)](#1), is an extension to the
k-means algorithm, which handles mixed numerical and categorical data.

Also, for completeness, note that a well-known Python implementation is
available [here](https://github.com/nicodv/kmodes).


## Design choices

...


### Public interface

A common framework for machine learning in Python is
[scikit-learn](https://scikit-learn.org/stable/index.html), which provides a
consistent [API](https://scikit-learn.org/stable/modules/classes.html) for
components. A natural choice would be therefore to implement the
[cluster mixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html).
However, mixed data is not easily represented using NumPy arrays. A common
workaround is to use a Pandas dataframe, as proposed by the experimental
[OpenML](https://openml.org/) support [function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html).

In order to tie input format to dataframes, k-prototypes methods accept two
arrays, a floating-point one and a integer one. Feature preprocessing,
including categorical values encoding, is not handled by the core algorithm.
Helpers are provided, though.


### Optimization strategy

Both [Huang (1997)](#1) and ...


### Cluster initialization

...


## References

<ol>
    <li id="1">
        Clustering large data sets with mixed numeric and categorical values,
        1997, Zhexue Huang
    </li>
    <li id="2">
        Extensions to the k-modes algorithm for clustering large data sets with
        categorical values, 1998, Zhexue Huang
    </li>
    <li id="3">
        A new initialization method for categorical data clustering, 2009,
        Fuyuan Cao, Jiye Liang, Liang Bai
    </li>
    <li id="4">
        A Novel Cluster Center Initialization Method for the k-Prototypes
        Algorithms using Centrality and Distance, 2015, Jinchao Ji, Wei Pang,
        Yanlin Zheng, Zhe Wang, Zhiqiang Ma and Libiao Zhang
    </li>
</ol>
