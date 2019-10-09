Iterative Closest Point
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* SOMANSHU AGARWAL
  * [LinkedIn](https://www.linkedin.com/in/somanshu25)
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Quadro P1000 4GB (Moore 100B Lab)

### Introduction

Iterative Closest Point (ICP) is one of the variants to implement scan matching on two pointclouds which are aligned at an angle and we want to overlap them. The algorithm find the transformation between a point cloud and some reference surface (or another point cloud), by minimizing the square errors between the corresponding entities.

### Theory Behind the Optimization:



### Different Implementation:

The fowllowing implementations are done to review the performance:
* CPU Naive Implementation
* GPU Naive Implementation
* KD Tree Implmentation in GPU

## CPU Implementation

CPU Implementation of the scan matching involves searching the correspondence point (the closest point) in the target pointcloud (reference) for each of the point in the source pointcloud (which needs to be transformed to target). The search is naive across all the points in the target pointcloud. After finding the correspondance, we are mean centering the source and correspondance points and then finding the SVD for the `XYT`. Then applying the theory mentioned above for finding the optimization, we calculate the rotation and translational matrix for each iteration of the alogorithm and update the source points.

## GPU Naive Implementation

In GPU Naive implementation, the difference with respect to the previous CPU Implementation is that there are `sourceSize` number of parallel threads which are released which compute the closest neighbour for each point in source pointcloud seperately. We cna perform this in parallel as each source point is independent of each other when finding the correspondance point in the target pointcloud. The implementation of mean centering, matirx multiplication and updating the source points are also done in parallel for each point in the source pointcloud to improve the time efficiency. For implementing the matrix multiplication, we used the concept of outer product to calculate the matrixes of each row and column vector pair and then apply the `thrust::reduce` to parallely add them to get the final multiplication result.

## KD-tree Implementation

In the previous implementations, we are doing the naive search over all the points in the target pointcloud  

