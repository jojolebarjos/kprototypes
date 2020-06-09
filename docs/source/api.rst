.. _api:


Developer Interface
===================

.. module:: kprototypes

This part of the documentation covers the public interface of k-prototypes.


Main Interface
--------------

The main entry point follows similar conventions to Scikit-Learn, but it is not
fully compatible (see design choices).

.. autoclass:: KPrototypes
   :members:
   :undoc-members:


Distance Measure
----------------

Common distance functions are provided, but any callable can be used, as long
as broadcasting is properly done.

.. autofunction:: check_distance
.. autofunction:: euclidean_distance
.. autofunction:: manhattan_distance
.. autofunction:: matching_distance


Initialization
--------------

Simple initialization functions are provided, but any callable can be used.
Explicit centroids can also be provided, either to resume training or to use
an external initialization process.

.. autofunction:: check_initialization
.. autofunction:: random_initialization
.. autofunction:: frequency_initialization


Data Preprocessing
------------------

As k-prototypes only accepts floating-point values for numerical data and
integer values for categorical data, any other data type must be properly
converted beforehand.

:py:class:`sklearn.preprocessing.StandardScaler` is the equivalent for
numerical values.

.. autoclass:: CategoricalTransformer
   :members:
   :undoc-members:


Low-Level Optimization Methods
------------------------------

At its core, k-prototypes is defined by these two methods.

.. autofunction:: fit
.. autofunction:: predict
