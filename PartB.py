# -*- coding: utf-8 -*-
"""
Michael Ventoso
MichaelVentoso@Gmail.com

ProtAtOnce Interview Phase 3
Part B: Cell Dataset
K-Means Clustering
"""
# Built-in
import os
import sys

# Libs
import numpy
import pandas
from scipy import spatial


"""
Creates a Pandas Dataframe for all the cell data and sets the column names
"""
def createCellDataFrame():

    #creates dataframe for first .csv file and sets column names
    cell_dataset = pandas.read_csv(os.path.join(sys.path[0], 'cell_dataset.csv'))

    cell_dataframe = pandas.DataFrame(cell_dataset,
                                       columns=['Name', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
                                                'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
                                                'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
                                                'x_16', 'x_17', 'x_18', 'x_19', 'x_20'])

    return cell_dataframe


"""
Calculates the upper and lower bounds of each cell feature
This is so our random starting points will be within the bounds of the cell data
"""
def calculateSpaceBounds(df):

    #removes name column so we don't compare strings with floats
    df = df.drop(['Name'], axis=1)

    #list of lists (1 pair of min/max bounds per cell feature)
    bounds = []

    #calculates and stores bounds
    for label, content in df.iteritems():
        bounds.append([content.min(), content.max()])

    return bounds


"""
Creates k number of random points (1 for each cluster)
Returns a dataframe of all the centers' feature values
"""
def createKPoints(k,bounds):

    #initialize dictionary to put points in
    dict_of_points = {}

    #loops through number of clusters and creates random point within the bounds of the cell data
    #also adds new point to dictionary
    for cluster in range(k):
        dimensions = []
        for feature in range(20):
            dimensions.append(numpy.random.uniform(bounds[feature][0], bounds[feature][1]))
        dict_of_points.update({ cluster : dimensions })

    #converts dictionary to dataframe
    centers = pandas.DataFrame(data=dict_of_points)
    return centers


"""
Given the index (name) of a cell from the dataset
Calculates and returns the nearest cluster based on Euclidean distance in 20D
"""
def findNearestCenter(df, centers, index):
    #calculates k and removes name column from cell dataframe
    k = len(centers.columns)
    df = df.drop(['Name'], axis=1)

    #I had an issue using the max value constant, so I compromised and used a large number (1E6)
    min_dist = 1000000.0
    nearest = -1

    #Loops through cluster's centerpoints to find the closest one
    for cluster in range(k):
        dist = spatial.distance.euclidean(df.loc[index].astype('float64'), centers[x])
        if dist < min_dist:
            nearest = cluster
            min_dist = dist

    return nearest

"""
Given a list of sets, where each set represents a cluster and is filled with the indexes of its cells
Recalculates and returns the new centers of each cluster
"""
def calculateCenters(df, set_list):

    #initializes number of clusters, dictionary of points, and removes name column from cell dataset
    num_clusters = len(set_list)
    dict_of_points = {}
    df = df.drop(['Name'], axis=1)

    #loops through clusters
    for cluster in range(num_clusters):

        #initialize data structures
        running_total = {}
        dimensions = []

        #determine number of cells in cluster
        num_cells = len(set_list[cluster])

        #loop through cells in cluster
        for cell in range(num_cells):

            #calculates the total sum of all features for cells in current cluster
            #sum of x_1, sum of x_2 etc.
            for feature in df.columns:
                if feature not in running_total:
                    running_total.update({feature: df.loc[cell, feature].astype('float64')})
                else:
                    running_total.update({feature: running_total.get(feature) + df.loc[cell, feature].astype('float64')})

        #divides totals by number of cells to get average of each feature for cells in current cluster
        #also deals with possibility of an empty cluster to avoid exception
        for feature2 in df.columns:
            if running_total.get(feature2) is None:
                running_total.update({feature2:0})

            if num_cells == 0:
                dimensions.append(0)
            else:
                dimensions.append(running_total.get(feature2)/num_cells)

        #updates dictionary with new centers
        dict_of_points.update({cluster: dimensions})

    #converts dictionary to dataframe
    centers = pandas.DataFrame(data=dict_of_points)
    return centers


"""
Does the K-Means Clustering Algorithm
Given the number of clusters (k) and the cell data
(also given trialNum for running multiple trials in a loop and the number of iterations to be performed)\

Each iteration:
    *FIRST TIME ONLY*- Randomly pick centers for each cluster
    
    Assign every cell to nearest cluster
    Recalculate clusters center
    Repeat for given number of iterations
"""
def kMeans(df,k,trialNum,iterations,centers=None):

    #calculate bounds for initial randomization of cluster centers
    bounds = calculateSpaceBounds(df)

    #only calculate cluster centers on the first pass
    if centers is None:
        centers = createKPoints(k,bounds)

    num_cells = len(df)

    set_list = []

    #main algorithm
    for i in range(iterations):

        print(i) #just to keep track of progress

        #sorts cells into cluster
        set_list = [set() for s in range(k)]
        for cell in range(num_cells):
            nearest = findNearestCenter(df,centers,cell)
            set_list[nearest].add(cell)

        #recalculates new cluster centers
        centers = calculateCenters(df, set_list, k)

    #transposes dataframe to match given .csv orientation
    df_return = pandas.DataFrame(list(set_list))
    df_return = df_return.transpose()
    df_return.to_csv('kMeans_k' + str(k) + '_n' + str(trialNum) + '.csv')


if __name__ == '__main__':
    print("STARTED")
    print()

    cell_dataframe = createCellDataFrame();

    #automatically do multiple trials of multiple K values
    for x in range(1, 6):
        for y in range(3, 8, 1):
            kMeans(cell_dataframe, y, x, 50)

    print("FINISHED")