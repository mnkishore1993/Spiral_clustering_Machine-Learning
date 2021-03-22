# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:19:00 2020

"""

import pandas as pd

import matplotlib.pyplot as mp

FourCircleFilePath = 'C:/Masters/Semester1/Machine Learing/Assignment2/FourCircle.csv'

dataset = pd.read_csv(FourCircleFilePath)



mp.scatter(dataset['x'],dataset['y'])
mp.xlabel('x values')
mp.ylabel('y values')
mp.title('Scatter plot x vs y')
mp.grid(True)
mp.show()

print('It appears to have 4 clusters')


import sklearn.cluster as cluster
kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(dataset[['x','y']])

dataset['Cluster'] = kmeans.labels_


mp.scatter(dataset['x'],dataset['y'],c=dataset['Cluster'])
mp.xlabel('x values')
mp.ylabel('y values')
mp.title('Scatter plot x vs y')
mp.grid(True)
mp.show()


import sklearn.neighbors as neighbors
import numpy
import math

trainData = dataset[['x','y']]

nObs = len(dataset)

kNNSpec = neighbors.NearestNeighbors(n_neighbors =10,
   algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
Adjacency = numpy.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = numpy.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

from numpy import linalg 
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = numpy.arange(1,7,1) 
mp.plot(sequence, evals[0:6,], marker = "o")
mp.xlabel('Sequence')
mp.ylabel('Eigenvalue')
mp.xticks(sequence)
mp.grid("both")
mp.show()

# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = numpy.arange(1,21,1) 
mp.plot(sequence, evals[0:20,], marker = "o")
mp.xlabel('Sequence')
mp.ylabel('Eigenvalue')
mp.grid("both")
mp.xticks(sequence)
mp.show()


import scipy.stats
# Inspect the values of the selected eigenvectors 
for j in range(4):
    print('Eigenvalue: ', j)
    print('              Mean = ', numpy.mean(evecs[:,j]))
    print('Standard Deviation = ', numpy.std(evecs[:,j]))
    print('  Coeff. Variation = ', scipy.stats.variation(evecs[:,j]))


Z = evecs[:,[0,1]]

mp.scatter(1e10*Z[:,0], Z[:,1])
mp.xlabel('First Eigenvector')
mp.ylabel('Second Eigenvector')
mp.grid("both")
mp.show()



kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=0).fit(Z)
dataset['SpectralCluster'] = kmeans_spectral.labels_

mp.scatter(dataset['x'], dataset['y'], c = dataset['SpectralCluster'])
mp.xlabel('x')
mp.ylabel('y')
mp.title('Scatter plot for x v/s y')
mp.grid(True)
mp.show()

