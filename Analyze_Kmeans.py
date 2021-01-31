# -*- coding: utf-8 -*-
"""
Michael Ventoso
MichaelVentoso@Gmail.com

ProtAtOnce Interview Phase 3
Part B: Cell Dataset
Analyzing K-Means Data
"""
# Built-in
import os
import sys

# Libs
import numpy
import pandas
import matplotlib.pyplot
from scipy import spatial


"""
Creates a Pandas Dataframe for all the cell data and sets the column names
"""
def createCellDataFrame():
    # creates dataframe for first .csv file and sets column names
    cell_dataset = pandas.read_csv(os.path.join(sys.path[0], 'cell_dataset.csv'))

    cell_dataframe = pandas.DataFrame(cell_dataset,
                                      columns=['Name', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
                                               'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
                                               'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
                                               'x_16', 'x_17', 'x_18', 'x_19', 'x_20'])

    return cell_dataframe


"""
Reads files outputted by PartB.py
Creates lists of sets that match formatting to the findNearestCenter in PartB
Each set is a cluster, and is filled with cell indexes
"""
def createClusterList(df, k, num_trial):

    #just in case
    if k <= 0 or num_trial <= -1:
        print("Number of clusters and number of trials must both be positive integers!")
        return

    #generates file string to read file
    file_string = 'kMeans_k' + str(k) + '_n' + str(num_trial + 1) + '.csv'

    #When running PartB I let the new files flood this directory, so I post-added them to a folder named KMeans_Data
    cluster_df = pandas.read_csv(os.path.join(sys.path[0], "KMeans_Data/" + file_string))

    #removes first column ('Names') and transposes to match orientation
    cluster_df = cluster_df.drop(cluster_df.columns[0], axis=1)
    cluster_df = cluster_df.transpose()

    #change the dataframe to a list
    cluster_list = cluster_df.values.tolist()

    set_list = [None] * k
    index = 0

    #dataframes can deal easily with missing data but every column must have the same number of rows and vice versa
    #therefore, since each cluster has a different number of cells, I need to remove the empty spaces
    for cluster in cluster_list:
        cluster = [cell for cell in cluster if not numpy.isnan(cell)]
        set_list[index] = set(cluster)
        index += 1

    return set_list


"""
Deals with the second part of this assignment
Given a specific trial
Returns the 5 most defining features of each cluster
Written after and based off of calculateAverageSpread
"""
def analyzeFeatures(df, k, numTrial):

    #removes name column from cell dataframe
    df = df.drop(['Name'], axis=1)
    spread_list = []

    #create cluster list and calculate centers
    cluster_list = createClusterList(df, k, numTrial)
    center = calculateCenter(df, cluster_list, k)
    cluster_spreads = []

    # for each cluster
    for cluster_id in range(k):
        num_cells = len(cluster_list[cluster_id])

        # for each point in each cluster
        for cell_id in cluster_list[cluster_id]:

            feature_spread = [0] * 20

            # keeps running total of difference between each feature and the center
            for feature in range(20):
                feature_spread[feature] += abs(df.loc[cell_id][feature].astype('float64') - center[cluster_id][feature])

        #divides total by the number of cells in the cluster
        #to calculate average spread of each feature to the cluster center across the cells in the cluster
        for x in range(20):
            feature_spread[x] = float("{0:.2f}".format(feature_spread[x] / num_cells))

        cluster_spreads.append(feature_spread)

    cluster_features = []

    #loops through each cluster
    for cluster in cluster_spreads:
        top_features = []

        #finds the five features with the lowest spread one at a time
        for y in range(5):
            index = cluster.index(min(cluster))
            top_features.append(index)
            cluster[index] = 100

        cluster_features.append(top_features)

    return cluster_features


"""
Since I don't save the locations of each clusters center (because it's always changing and can be calculated)
I need this function to figure out the centers of each cluster.  It is a slightly modified version from PartB
"""
def calculateCenter(df, set_list, k):
    num_clusters = len(set_list)
    dict_of_points = {}

    for x in range(num_clusters):
        running_total = {}
        dimensions = []

        num_cells = len(set_list[x])

        #loop through cells in cluster
        for cell in range(num_cells):

            # calculates the total sum of all features for cells in current cluster
            # sum of x_1, sum of x_2 etc.
            for feature in df.columns:

                if feature not in running_total:
                    running_total.update({feature: df.loc[cell, feature].astype('float64')})
                else:
                    running_total.update({feature: running_total.get(feature) + df.loc[cell, feature].astype('float64')})

        # divides totals by number of cells to get average of each feature for cells in current cluster
        # also deals with possibility of an empty cluster to avoid exception
        for q in df.columns:
            if running_total.get(q) is None:
                running_total.update({q: 0})

            if num_cells == 0:
                dimensions.append(0)
            else:
                dimensions.append(running_total.get(q) / num_cells)

        dict_of_points.update({x: dimensions})

    #converts dictionary to dataframe
    centers = pandas.DataFrame(data=dict_of_points)

    return centers


"""
Given an k value and the number of trials there is data for
Calculates the average spreads for clusters in a given trial

Can be easily modified to give raw spread data
"""
def calculateAverageSpread(df, k, numTrials):

    #remove name column from dataframe
    df = df.drop(['Name'], axis=1)
    spread_list = []

    # for each trial
    for trial_id in range(numTrials):
        trial_list = []
        cluster_list = createClusterList(df, k, trial_id)
        centers = calculateCenter(df, cluster_list, k)

        # for each cluster
        for cluster_id in range(k):
            num_cells = len(cluster_list[cluster_id])
            cluster_spread = 0

            # for each cell in each cluster
            for cell_id in cluster_list[cluster_id]:
                cell_spread = 0

                # for each feature of each cell
                # gets total spread
                # by summing distance between cell's feature value and center's feature value for all features
                for x in range(20):
                    cell_spread += spatial.distance.euclidean(df.loc[cell_id].astype('float64'), centers[cluster_id])

                #adds individual cells spread to get total spread of the cluster
                cluster_spread += cell_spread

            #calculates average spread for each cluster
            trial_list.append(cluster_spread / num_cells)

        #calculates average spread for each trial
        spread_list.append(sum(trial_list) / k)

    return spread_list



"""
Plots data generated by calculating average spreads
"""
def visualizeData(data_list, numTrials):
    average_list = []
    total_list = [item for sublist in data_list for item in sublist]
    x_axis = []
    n = len(data_list)

    #formats data for plotting
    for x in range(numTrials):
        average_list.append(sum(data_list[x]) / numTrials)
        for y in range(numTrials):
            x_axis.append(3 + x)

    #prints data since it is a small amount
    print("Total List")
    print(total_list)
    print("Average List")
    print(average_list)

    #plots a scatter of each trials average cluster spread
    #and plots the average cluster spread for each K value- connected by lines
    matplotlib.pyplot.scatter(x_axis, total_list)
    matplotlib.pyplot.plot(range(3, 8), average_list, '-o')

    # sets labels for plot
    matplotlib.pyplot.suptitle("Average Spread vs. Number of Clusters")
    matplotlib.pyplot.xlabel("Number of Clusters")
    matplotlib.pyplot.ylabel("Average Spread")

    # displays and saves the scatterplot
    matplotlib.pyplot.savefig("KMeansPlot.png")
    matplotlib.pyplot.show()


if __name__ == '__main__':
    print("STARTED")
    print()
    numTrials = 5

    df = createCellDataFrame()

    trial_average_spreads = []
    for x in range(3,8):
        trial_average_spreads.append(calculateAverageSpread(df, x, numTrials))
    print(trial_average_spreads)
    visualizeData(trial_average_spreads, numTrials)

    print(analyzeFeatures(df, 5, 1))

    print("FINISHED")
