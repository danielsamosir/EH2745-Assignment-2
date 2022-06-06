#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:58:11 2022

@author: Daniel
"""

from k_means import KMeans
from kNN import kNN
import pandas as pd
import numpy as np
from global_ import accuracy
from functools import partial
import tkinter

def run_k_means():
    # Get data
    voltages = pd.read_excel('dataset_norm_labeled.xlsx')

    # Separating the DataFrame into a Learning set and a Test set
    # Select only the input columns from the dataframe and convert it to a Numpy
    # array to speed up calculations
    dataset_labeled = voltages.iloc[:, 1:]
    dataset = dataset_labeled.drop(['state'], axis=1).to_numpy()

    # For clustering k-means
    inputs = dataset
    # The outputs data is the state columns from the voltages dataframe
    outputs = voltages['state'].to_numpy()

    clusters = len(np.unique(outputs))
    print("Total of clusters defined in dataset: ", clusters)
    k = KMeans(max_iters=100)
    y_pred, ks = k.predict(inputs)
    print("Number of clusters based on k-means:", ks)
    print("Contigency Table or Cross Table of 'True' Outputs and Prediction: ")
    print(pd.crosstab(outputs, y_pred))
    

def run_KNN(k):
    no_k = int(k)
    # Get data
    voltages = pd.read_excel('dataset_norm_labeled.xlsx')

    # Separating the DataFrame into a Learning set and a Test set
    # Select only the input columns from the dataframe and convert it to a Numpy
    # array to speed up calculations
    dataset_labeled = voltages.iloc[:, 1:]
    dataset = dataset_labeled.drop(['state'], axis=1).to_numpy()

    # For clustering k-means
    inputs = dataset
    # The outputs data is the state columns from the voltages dataframe
    outputs = voltages['state'].to_numpy()

    n_time_steps = 60
    train_coefficient = 0.8
    n_training = int(train_coefficient * n_time_steps)

    # Separating the learning set from the test set
    inputs_LS_labeled = pd.concat([dataset_labeled[:n_training],
                                            dataset_labeled[n_time_steps:n_training + n_time_steps],
                                            dataset_labeled[2 * n_time_steps:n_training + 2 * n_time_steps],
                                            dataset_labeled[3 * n_time_steps:n_training + 3 * n_time_steps],
                                            dataset_labeled[4 * n_time_steps:n_training + 4 * n_time_steps],
                                            dataset_labeled[5 * n_time_steps:n_training + 5 * n_time_steps],
                                            dataset_labeled[6 * n_time_steps:n_training + 6 * n_time_steps]],
                                            axis=0, ignore_index=True)

    inputs_TS_labeled = pd.concat([dataset_labeled[n_training:n_time_steps],
                                            dataset_labeled[n_training + n_time_steps:2 * n_time_steps],
                                            dataset_labeled[n_training + 2 * n_time_steps:3 * n_time_steps],
                                            dataset_labeled[n_training + 3 * n_time_steps:4 * n_time_steps],
                                            dataset_labeled[n_training + 4 * n_time_steps:5 * n_time_steps],
                                            dataset_labeled[n_training + 5 * n_time_steps:6 * n_time_steps],
                                            dataset_labeled[n_training + 6 * n_time_steps:7 * n_time_steps]],
                                            axis=0, ignore_index=True)

    inputs_LS = inputs_LS_labeled.drop(['state'], axis=1).to_numpy()
    outputs_LS = inputs_LS_labeled['state'].to_numpy()
    inputs_TS = inputs_TS_labeled.drop(['state'], axis=1).to_numpy()
    outputs_TS = inputs_TS_labeled['state'].to_numpy()

    clf = kNN(k=no_k)
    clf.fit(inputs_LS, outputs_LS)
    predictions = clf.predict(inputs_TS)
    print("Prediction: ", predictions)
    print("Test data: ", outputs_TS)
    print("Accuracy of classification: ", accuracy(outputs_TS, predictions), "%")    


# GUI
def open_k_means():
    window_kmeans = tkinter.Toplevel()
    window_kmeans.title("Power system operating state clustering using k-means")
    window_kmeans.geometry("800x200")
    button_run_kmeans = tkinter.Button(window_kmeans, text="Run", command=run_k_means).pack()   

def open_KNN():
    window_KNN = tkinter.Toplevel()
    window_KNN.title("Power system operating state classification using K Nearest Neighbour")
    window_KNN.geometry("800x300")    
    tkinter.Label(window_KNN, text="Please enter the number of k :").pack()
    button_run_KNN1 = tkinter.Button(window_KNN, text="Run with k = 1", command=partial(run_KNN, "1")).pack()
    button_run_KNN2 = tkinter.Button(window_KNN, text="Run with k = 2", command=partial(run_KNN, "2")).pack()
    button_run_KNN3 = tkinter.Button(window_KNN, text="Run with k = 3", command=partial(run_KNN, "3")).pack()
    button_run_KNN4 = tkinter.Button(window_KNN, text="Run with k = 4", command=partial(run_KNN, "4")).pack()
    button_run_KNN5 = tkinter.Button(window_KNN, text="Run with k = 5", command=partial(run_KNN, "5")).pack()
    button_run_KNN6 = tkinter.Button(window_KNN, text="Run with k = 6", command=partial(run_KNN, "6")).pack()
    button_run_KNN7 = tkinter.Button(window_KNN, text="Run with k = 7", command=partial(run_KNN, "7")).pack()     
    button_run_KNN8 = tkinter.Button(window_KNN, text="Run with k = 8", command=partial(run_KNN, "8")).pack()
    button_run_KNN9 = tkinter.Button(window_KNN, text="Run with k = 9", command=partial(run_KNN, "9")).pack()        
    button_run_KNN10 = tkinter.Button(window_KNN, text="Run with k = 10", command=partial(run_KNN, "10")).pack()                  


window = tkinter.Tk()
window.title("EH2745 - Assignment 2: Clustering & Classification")
window.geometry("400x200")
label = tkinter.Label(window, text="Select algorithm to be run:").pack()
button1 = tkinter.Button(window, text="k-means clustering", command=open_k_means).pack()
button2 = tkinter.Button(window, text="KNN classification", command=open_KNN).pack()
window.mainloop()