
# K-prototypes

K-prototypes, as introduced by [Huang (1997)](#1), is an extension to the
k-means algorithm, which handles mixed numerical and categorical data.

Also, for completeness, note that a well-known Python implementation is
available [here](https://github.com/nicodv/kmodes).


## Installation

Install from source, to ensure latest version:

```
pip install git+https://gitlab.com/jojolebarjos/kprototypes.git
```

`fastkde` is an optional dependencies, required by the frequency-based initialization:

```
pip install cython numpy scipy
pip install fastkde
```


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


## Changelog

 * 0.1.1 - 2020-06-03
    * Add clean initialization procedures
 * 0.1.0 - 2020-05-04
    * Initial version
