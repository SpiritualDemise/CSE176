The Laplacian K-modes algorithm for clustering

(C) 2015 by Weiran Wang and Miguel A. Carreira-Perpinan
    Electrical Engineering and Computer Science
    University of California, Merced
    http://eecs.ucmerced.edu

This Matlab code implements the Laplacian K-modes algorithm described in
this paper:

  Wang, W. and Carreira-Perpiñán, M. Á. (2014): "The Laplacian K-modes
  algorithm for clustering". Jun. 15, 2014, arXiv:1406.3895 [cs.LG].

Typically, the user sets the following parameters: the number of clusters K, 
the bandwidth of the kernel density estimate s, and the trade-off parameter 
for smoothing lambda. Laplacian K-modes partitions the dataset into K 
soft clusters, each of which is represented by a mode of its density. In 
contrast, K-means partitions the dataset into hard K clusters, each of which
is represented by its mean.

Demo files (and datasets) that demonstrate the usage of lapkmodes.m with
some datasets used in the paper:
- demo_2moons.m: demonstrates the usage of Laplacian K-modes (using homotopy)
  and its out-of-sample mapping on the 2-moons dataset (file 2moons.mat).
- demo_5spirals.m: 5-spirals dataset.

List of functions:
- lapkmodes: the Laplacian K-modes clustering algorithm.
- lapkmodes_map: Laplacian K-modes out-of-sample mapping, predicts cluster
  assignments for a test point.
- SimplexProj: projection onto the probability simplex, used in the
  out-of-sample mapping of Laplacian K-modes.
- kmeans: the K-means clustering algorithm.
- sqdist: matrix of Euclidean squared distances between two point sets.
- gaussaff: build a k-nearest-neighbor or epsilon-ball graph on dataset.
- lass: trains a LASS model, given item-item and item-category affinities.
  This function is only necessary if using ADMM to train Laplacian K-modes.

