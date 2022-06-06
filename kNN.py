#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:58:22 2022

@author: Daniel
"""

import numpy as np
import pandas as pd
from collections import Counter
from global_ import euclidean_distance
import matplotlib.pyplot as plt

class kNN:

    def __init__(self, k=7):
        """
        Initialize KNN class that will run classification algorithm
        Input: no of k
        Output: None
        """             
        self.k = k

    def fit(self, X, y):
        """
        Fitting the features and label learning data 
        Input: Features learning data, Label learning data
        Output: None
        """             
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the label of input test based on known learning data 
        Input: input test data
        Output: output test data (label)
        """  
        predicted_labels = [self._predict(x) for x in X]

		# plot bus1
        training_bus1 = self.X_train[:, [0,9]]        
        input_bus1 = X[1, [0,9]].T
        self.plot(training_bus1, input_bus1, "Bus 1")
        # plot bus2
        training_bus2 = self.X_train[:, [1,10]]        
        input_bus2 = X[1, [1,10]].T
        self.plot(training_bus2, input_bus2, "Bus 2") 
        # plot bus3
        training_bus3 = self.X_train[:, [2,11]]        
        input_bus3 = X[1, [2,11]].T
        self.plot(training_bus3, input_bus3, "Bus 3")  
        # plot bus4
        training_bus4 = self.X_train[:, [3,12]]        
        input_bus4 = X[1, [3,12]].T
        self.plot(training_bus4, input_bus4, "Bus 4")  
        # plot bus5
        training_bus5 = self.X_train[:, [4,13]]        
        input_bus5 = X[1, [4,13]].T
        self.plot(training_bus5, input_bus5, "Bus 5")  
        # plot bus6
        training_bus6 = self.X_train[:, [5,14]]        
        input_bus6 = X[1, [5,14]].T
        self.plot(training_bus6, input_bus6, "Bus 6")  
        # plot bus7
        training_bus7 = self.X_train[:, [6,15]]        
        input_bus7 = X[1, [6,15]].T
        self.plot(training_bus7, input_bus7, "Bus 7")  
        # plot bus8
        training_bus8 = self.X_train[:, [7,16]]        
        input_bus8 = X[1, [7,16]].T
        self.plot(training_bus8, input_bus8, "Bus 8")  
        # plot bus9
        training_bus9 = self.X_train[:, [8,17]]        
        input_bus9 = X[1, [8,17]].T
        self.plot(training_bus9, input_bus9, "Bus 9")  

        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Compute distances input to all learning data and specify label of the input
        Input: current input data
        Output: label of current input data
        """          
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Voting
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]

    def plot(self, train, input, title):
        """
        Plotting 2-d data of each bus (only for first test input data)
        Input: learning data, test input data, Bus Name
        Output: None
        """        
        fig, ax = plt.subplots(figsize=(15, 10))

        for point in train:
            ax.scatter(*point)

        ax.scatter(*input, marker="x", linewidth=2, s=1000)

        plt.xlabel("Voltage Magnitude (normalized, in %)")
        plt.ylabel("Voltage Angle (normalized, in %)")
        plt.title(title)
        plt.show()        