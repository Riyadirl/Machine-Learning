import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Load data from jain_feats.txt into a 2D numpy array X
X = np.loadtxt('jain_feats.txt')

# Load initial centroids from jain_centers.txt into a 2D numpy array centroid_old
centroid_old = np.loadtxt('jain_centers.txt')

# Initialize centroid_new array with zeros
centroid_new = np.zeros_like(centroid_old)

# Iterate for 1000 epochs
for e in range(1000):

    # Assign points to centroids/clusters
    label = np.zeros(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        dist = np.linalg.norm(x - centroid_old, axis=1)
        label[i] = np.argmin(dist)

    # Update centroids
    for j in range(centroid_new.shape[0]):
        centroid_new[j] = np.mean(X[label == j], axis=0)

    # Check stop condition
    diff = np.max(np.abs(centroid_new - centroid_old))
    if diff < 1E-7:
        break
    else:
        centroid_old = centroid_new.copy()

# Final cluster centroids
print(centroid_old)
