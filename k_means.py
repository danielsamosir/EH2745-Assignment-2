#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:58:27 2022

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from global_ import euclidean_distance
from global_ import Clusters

class KMeans:
    def __init__(self, max_iters=100):
        """
        Initialize KMeans class that will run clustering algorithm
        Input: no of maximum iteration for elbow method and finding local optima
        Output: None
        """            
        self.max_iters = max_iters
        # list of sample indices for each cluster
        self.clusters = []
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        """
        Initialize KMeans class that will run the clustering algorithm
        Input: Input data
        Output: result data shown in label, no of clusters
        """            
        self.X = X
        self.n_samples, self.n_features = X.shape
        # Elbow method:
        j_diff_threshold = 45 #Manually tuned
        diff_k_clusters = []
        for k in range(1,8):
            
            # Finding Local optima through iteration:
            clusters = []
            for i in range(self.max_iters):
                # Initialize
                random_sample_idxs = np.random.choice(self.n_samples, k, replace=False)
                self.centroids = [self.X[idx] for idx in random_sample_idxs]
                # list of sample indices for each cluster
                self.clusters = [[] for _ in range(k)]

                # Iterate to get optimized clusters
                for _ in range(self.max_iters):
                    # Assign samples to closest centroids (create clusters)
                    self.clusters = self._create_clusters(self.centroids, k)

                    # Calculate new centroids from the clusters
                    centroids_old = self.centroids
                    self.centroids = self._get_centroids(self.clusters, k)

                    # check if clusters have changed
                    if self._is_converged(centroids_old, self.centroids, k):
                        break

                # Classify samples as the index of their clusters     
                result = self._get_cluster_labels(self.clusters)
                j = self._j_metric(self.clusters, self.centroids)
                new_clusters = Clusters(i, k, self.centroids, self.clusters, result, j)
                clusters.append(new_clusters)

            min_j_idx = Clusters.lowest_j_idx
            
            #return clusters[min_j_idx].label
            diff_k_clusters.append(clusters[min_j_idx])

            if k > 1:
                # Calculate j metrics improvement
                j_diff = diff_k_clusters[k-2].j - diff_k_clusters[k-1].j
                if j_diff <= j_diff_threshold:
                    #plot bus1
                    centroids_bus1 = np.array(diff_k_clusters[k-1].centroids)[:, [0,9]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus1, 0, 9, "Bus 1")
                    #plot bus2
                    centroids_bus2 = np.array(diff_k_clusters[k-1].centroids)[:, [1,10]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus2, 1, 10, "Bus 2")
                    #plot bus3
                    centroids_bus3 = np.array(diff_k_clusters[k-1].centroids)[:, [2,11]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus3, 2, 11, "Bus 3")
                    #plot bus4
                    centroids_bus4 = np.array(diff_k_clusters[k-1].centroids)[:, [3,12]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus4, 3, 12, "Bus 4")
                    #plot bus5
                    centroids_bus5 = np.array(diff_k_clusters[k-1].centroids)[:, [4,13]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus5, 4, 13, "Bus 5")
                    #plot bus6
                    centroids_bus6 = np.array(diff_k_clusters[k-1].centroids)[:, [5,14]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus6, 5, 14, "Bus 6")
                    #plot bus7
                    centroids_bus7 = np.array(diff_k_clusters[k-1].centroids)[:, [6,15]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus7, 6, 15, "Bus 7")                    
                    #plot bus8
                    centroids_bus8 = np.array(diff_k_clusters[k-1].centroids)[:, [7,16]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus8, 7, 16, "Bus 8")  
                    #plot bus9
                    centroids_bus9 = np.array(diff_k_clusters[k-1].centroids)[:, [8,17]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus9, 8, 17, "Bus 9")  

                    return diff_k_clusters[k-1].label, k

    def _create_clusters(self, centroids, k):
        """
        Assign the samples to the closest centroids to create clusters
        Input: centroids values of all features, no of k
        Output: indices of all samples per each centroid/k
        """             
        clusters = [[] for _ in range(k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Finding distance of the current sample to each centroid
        Input: sample data, centroids data
        Output: index of closest centroid
        """            
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters, k):
        """
        Finding new centroids by assigning mean value of clusters to centroids
        Input: indices of all samples per each centroid/k, no of k
        Output: new centroids
        """              
        centroids = np.zeros((k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids, k):
        """
        Comparing distances between each old and new centroids, for all centroids
        Input: old centroids, new centroids, no of k
        Output: Boolean(Converged or Not Converged)
        """         
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(k)
        ]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        """
        Assigning each sample to label of defined cluster
        Input: indices of all samples per each centroid/k
        Output: labels of all samples (result)
        """             
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _j_metric(self, clusters, centroids):
        """
        Calculate the total distance of all samples to their centroids
        Input: indices of all samples per each centroid/k, centroids
        Output: j metric
        """         
        j = 0
        for cluster_idx, k in enumerate(clusters):
            centroid = centroids[cluster_idx]
            for idx in k:
                sample = self.X[idx]
                j += euclidean_distance(sample, centroid)
        return j
    
    def plot(self, clusters, centroids, col1, col2, title):
        """
        Plotting 2-d data of each bus
        Input: clusters, centroids, VoltMag colomn, VoltAng colomn, Bus Name
        Output: None
        """            
        fig, ax = plt.subplots(figsize=(15, 10))

        for i, index in enumerate(clusters):
            point = np.array(self.X[index])[:,[col1, col2]].T
            ax.scatter(*point)

        for point in centroids:
            ax.scatter(*point, marker="x", linewidth=2)

        plt.xlabel("Voltage Magnitude (normalized, in %)")
        plt.ylabel("Voltage Angle (normalized, in %)")
        plt.title(title)
        plt.show()