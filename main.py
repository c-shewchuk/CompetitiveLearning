"""
# CMPE 452 Assignment 3
## Kohonen and kMeans Competitive Neural Networks
### Curtis Shewchuk
### SN: 10189026

The main file which will be run. Where the training algorithms are called.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import comp as cp

## Import and shuffle the data

filename = 'dataset_noclass.csv'

raw = pd.read_csv(filename, skiprows=1)
data = np.array(raw)

## Generate all necessary weights and cluster arrays
rows = len(data[:,0])
weightsKohonen = np.random.rand(2,3)
clusterKohonen = np.zeros(rows)
clusterKMeans = np.zeros(rows)
weightsKMeans = 2*np.random.rand(2,3)
learningRate = 0.5
maxEpochs = 10
aFile = open('outputs.txt', 'w')

## Train the networks

weightsKohonen = cp.trainKohonen(data, weightsKohonen, maxEpochs, rows, learningRate)

weightsKMeans, errorOneKmeans, errorTwoKmeans = cp.trainKMeans(data, weightsKMeans, maxEpochs, clusterKMeans)

# Create Plots
plt.figure(1)
plt.scatter(data[:,0],data[:,1])
plt.scatter(weightsKohonen[0,0],weightsKohonen[0,1],s = 200,c = 'r')
plt.scatter(weightsKohonen[1,0],weightsKohonen[1,1], s = 200, c = 'k')
plt.title('Kohonen Output')
plt.show()

plt.figure(2)
plt.scatter(data[:,0], data[:,1])
plt.scatter(weightsKMeans[0,0],weightsKMeans[0,1],s = 200,c = 'r')
plt.scatter(weightsKMeans[1,0],weightsKMeans[1,1], s = 200, c = 'k')
plt.title("KMeans Output")
plt.show()

# Write to Text file
aFile.write("CMPE 452 Assignment 3\nKohonen and kMeans Competitive Neural Networks\nCurtis Shewchuk\nSN: 10189026\n")
aFile.write('The Final Weights:\n')
aFile.write('Kohonen\n')
aFile.write("".join(str(elem) for elem in weightsKohonen))
aFile.write('\nKmeans\n')
aFile.write(''.join(str(elem)for elem in weightsKMeans))
aFile.write('\n\n')
aFile.write('Errors\n')
aFile.write('kMeans\n')
aFile.write('Error One: ' + str(errorOneKmeans) + " Error Two: " + str(errorTwoKmeans))

print("Run Complete")

