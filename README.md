
# K-prototypes

K-prototypes, as introduced by [Huang (1997)](#ref1), is an extension to the
k-means algorithm, which handles mixed numerical and categorical data.

Also, for completeness, note that a well-known Python implementation is
available [here](https://github.com/nicodv/kmodes).


## Installation

Install from source, to ensure latest version:

```
pip install git+https://github.com/jojolebarjos/kprototypes.git
```

Some examples require [UMAP](https://github.com/lmcinnes/umap) for dimensionality reduction and [matplotlib](https://matplotlib.org/) for rendering:

```
pip install matplotlib umap-learn
```


## References

<ol>
    <li><a name="ref1"></a>
        Clustering large data sets with mixed numeric and categorical values,
        1997, Zhexue Huang
    </li>
    <li><a name="ref2"></a>
        Extensions to the k-modes algorithm for clustering large data sets with
        categorical values, 1998, Zhexue Huang
    </li>
    <li><a name="ref3"></a>
        A new initialization method for categorical data clustering, 2009,
        Fuyuan Cao, Jiye Liang, Liang Bai
    </li>
    <li><a name="ref4"></a>
        A Novel Cluster Center Initialization Method for the k-Prototypes
        Algorithms using Centrality and Distance, 2015, Jinchao Ji, Wei Pang,
        Yanlin Zheng, Zhe Wang, Zhiqiang Ma and Libiao Zhang
    </li>
</ol>


## Changelog

 * 0.1.3 - 2024-03-30
    * Migrated to GitHub
 * 0.1.2 - 2020-12-04
    * Add proper documentation
    * Small fixes
 * 0.1.1 - 2020-06-03
    * Add clean initialization procedures
 * 0.1.0 - 2020-05-04
    * Initial version
