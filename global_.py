#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:00:14 2022

@author: Daniel
"""

import numpy as np

# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Accuraccy of classification
def accuracy(outputs_TS, predictions):
    return 100*np.sum(outputs_TS == predictions) / len(outputs_TS)

# Class for storing attributes (including class attributes) while finding the lowest j metric 
class Clusters:
    lowest_j = None
    lowest_j_idx = None

    def __init__(self, i, k, centroids, sample_idx, label, j):
        self.i = i
        self.k = k
        self.centroids = centroids
        self.sample_idx = sample_idx
        self.label = label
        self.j = j

        if (Clusters.lowest_j == None) or (j < Clusters.lowest_j):
            Clusters.lowest_j = j
            Clusters.lowest_j_idx = i
        

