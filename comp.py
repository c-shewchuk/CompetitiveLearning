"""
# CMPE 452 Assignment 3
## Kohonen and kMeans Competitive Neural Networks
### Curtis Shewchuk
### SN: 10189026

Functions required to implement both Kohonen and KMeans learning algorithms.
"""
import numpy as np


def findWinnerNode(inputs, weights):
    """
    Determines which node was the closest node to the centroid for Kohonen.
    :param inputs:
    :param weights:
    :return:
    """

    distanceOne = np.sqrt(np.sum((inputs-weights[0,:])**2))
    distanceTwo = np.sqrt(np.sum((inputs-weights[1,:])**2))
    if distanceTwo <= distanceOne:
        return 1
    else:
        return 0

def updateWeights(inputs, weights, winningNode, learnRate):
    """
    Updates the centroid weights for the Kohonen network.
    :param inputs:
    :param weights:
    :param winningNode:
    :param learnRate:
    :return:
    """
    return weights[winningNode,:]+learnRate*(inputs - weights[winningNode, :])

def updateLearningRate(oldRate):
    """
    Updates our learning rate by a constant value. Currently multiplies old by 0.8.
    :param oldRate: The learning rate previously used.
    :return: Newly calculated learning rate.
    """
    return oldRate*0.8

def chooseRandomInput(inputData):
    """
`   Randomly selects a data row from the shuffled input data.
    :param inputData:
    :return:
    """
    return inputData[np.random.randint(0,len(inputData[:,0]),size=1),:]


def trainKohonen(inputs, weights, maxEpochs, rows, learningRate):
    """
    Train a simple network to the Kohonen competitive network.
    :param inputs:
    :param weights:
    :param maxEpochs:
    :param rows:
    :param learningRate:
    :return weights:
    """
    for i in range(maxEpochs):
        for j in range(rows):
            input = chooseRandomInput(inputs)
            winner = findWinnerNode(input, weights)
            weights[winner, :] = updateWeights(input, weights, winner, learningRate)
            learningRate = updateLearningRate(learningRate)

    return weights


##### KMEANS FUNCTIONS #####

def calculateDistance(inputs, weights):
    """
    Calculates the distance to the centroid
    :param inputs:
    :param weights:
    :return:
    """
    distanceOne = np.abs(inputs-weights[0,:])
    distanceTwo = np.abs(inputs-weights[1,:])
    return distanceOne, distanceTwo

def determineCluster(inputs, weights, clusters):
    """
    Determines which element the cluster will be in
    :param inputs:
    :param weights:
    :param clusters:
    :return:
    """
    distanceOne, distanceTwo = calculateDistance(inputs, weights)
    magnitudeOne = np.sqrt(np.sum(distanceOne**2, axis=1))
    magnitudeTwo = np.sqrt(np.sum(distanceTwo**2, axis=1))
    difference = np.array(magnitudeTwo - magnitudeOne)
    #Assign calculations to clusters
    clusters[difference>=0] = 0
    clusters[difference<0] = 1
    ## check for Errors
    errorOne = np.sum(magnitudeOne[(difference-1).astype(bool)])
    errorTwo = np.sum(magnitudeTwo[difference.astype(bool)])

    return clusters, errorOne, errorTwo

def findCentroids(inputs, weights, clusters):
    """
    Finds the centroids.
    :param inputs:
    :param weights:
    :param clusters:
    :return:
    """
    dataClusterOne = inputs[(clusters-1).astype(bool)]
    dataClusterTwo = inputs[clusters.astype(bool)]

    ## Calculate averages to find the centroids
    weights[0,:] = np.sum(dataClusterOne, axis=0)/len(dataClusterOne[:,0])
    weights[1,:] = np.sum(dataClusterTwo, axis=0)/len(dataClusterTwo[:,0])

    return weights[0,:], weights[1,:]

def trainKMeans(inputs, weights, maxEpochs, clusters):
    """
    Trains the network using the KMeans algorithm.
    :param inputs:
    :param weights:
    :param maxEpochs:
    :param rows:
    :param learningRate:
    :return:
    """
    errorThreshold = 10
    for index in range(maxEpochs):
        clusters, errorOne, errorTwo = determineCluster(inputs, weights, clusters)

        if (errorOne + errorTwo) < errorThreshold:
             break

        weights[0,:], weights[1,:] = findCentroids(inputs, weights, clusters)

    return weights, errorOne, errorTwo