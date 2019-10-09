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

CPU Implementation of the scan matching involves searching the correspondence point (the closest point) in the target pointcloud (reference) for each of the point in the source pointcloud (which needs to be transformed to target). The search is naive across all the points in the target pointcloud. After finding the correspondance, we are mean centering the source and correspondance points and then finding the SVD for the `XYT`. 

