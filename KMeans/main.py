import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Load data from jain_feats.txt into a 2D numpy array X
X = np.loadtxt('jain_feats.txt')

# Load initial centroids from jain_centers.txt into a 2D numpy array centroid_old
centroid_old = np.loadtxt('jain_centers.txt')
centroid_new = np.zeros_like(centroid_old)

# Initialize label array
label = np.zeros(X.shape[0])

# Define a function to calculate Euclidean distance


def distance(x, y):
    return np.sqrt(np.sum((x-y)**2))


# Run k-means algorithm
max_dif = 1E-7
while True:
    # Assign points to centroids
    for i in range(X.shape[0]):
        dist = np.zeros(centroid_old.shape[0])
        for j in range(centroid_old.shape[0]):
            dist[j] = distance(X[i], centroid_old[j])
        label[i] = np.argmin(dist)

    # Update centroids
    for j in range(centroid_new.shape[0]):
        centroid_new[j] = np.mean(X[label == j], axis=0)

    # Check stop condition
    diff = np.max(np.abs(centroid_new - centroid_old))
    if diff < max_dif:
        break

    # Move to next iteration
    centroid_old = centroid_new.copy()
