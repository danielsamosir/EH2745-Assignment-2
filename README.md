# EH2745-Assignment-2
Clustering (k-means) and classification (KNN) of timeseries power system measurement

The purpose of assingnment 2 is to combine machine learning techniques with timeseries power system modelling in pandapower. One supervised learning algorithms (KNN) and one unsupervised learing algorithm (k-means) will be implemented from scratch to predict the operating state of power system depending on the measurement of voltage magnitude and voltage angle of all the bus. The output is labels of operating states in which the power system operated.

The single line diagram of power system under study is shown below:
![sld](https://user-images.githubusercontent.com/33414239/172237366-78c9a788-d11b-4cbd-b8a2-e6eca24fad89.png)
- 9 Buses
- 1 Slack Bus
- 3 Generator Buses
- 3 Load Buses
- All bus are 110 kV
- All lines are 10 km length each
- All lines are configured using standard type: "149-AL1/24-ST1A"

There are 7 operating states specified in the dataset_generation.py:
- Normal load: P and Q is set as nominal value, and add some noise with a standard deviation of about 5-10% of the nominal values.
- High load: Set the P and Q for each load to 120% of nominal value, and add some noise with a standard deviation of about 5-10% of the nominal values.
- Low load: Set the P and Q for each load to 75% of nominal value, and add some noise with a standard deviation of about 5-10% of the nominal values.
- Generator 3 disconnected during high load: Use high load profile, disconnect generator 3
- Generator 3 disconnected during low load: Use low load profile, disconnect generator 3
- Line bus5-6 disconnected during high load: Use high load profile, disconnect line bus5-6
- Line bus5-6 disconnected during low load: Use low load profile, disconnect line bus5-6

All the results from simulated timeseries power flow will then be stored in two excel files:
- dataset.xlxs (all dataset shown in pu for voltage magnitude and degree in voltage angle)
- dataset_norm_labeled.xlxs (normalized dataset with label, will be used for the machine learning algorithms)

For k-means: 
All dataset will be used as input

For KNN:
dataset_norm_labeled will be splitted into training data (80%) and test data (20%).

To use the code, run main.py. A GUI window will show up for the user to select between two algorithms. For KNN, you have to select the number of k used in the calculation. Note that the k-means algorithm could take some time to run. The code then will show the plot of 2-dimensions data of each bus. You have to close the figure to make the code continue to the next step.

The implementation of k-means and K nearest neighbour algoritms is inspired by Youtuber: Python Engineer (https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E)
