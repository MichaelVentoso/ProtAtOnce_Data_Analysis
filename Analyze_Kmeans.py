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


def createCellDataFrame():
    # creates dataframe for first .csv file and sets column names
    cell_dataset = pandas.read_csv(os.path.join(sys.path[0], 'cell_dataset.csv'))

    cell_dataframe = pandas.DataFrame(cell_dataset,
                                      columns=['Name', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
                                               'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
                                               'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
                                               'x_16', 'x_17', 'x_18', 'x_19', 'x_20'])

    return cell_dataframe


def createClusterList(df, k, num_trial):
    if k <= 0 or num_trial <= -1:
        print("Number of clusters and number of trials must both be positive integers!")
        return

    file_string = 'kMeans_k' + str(k) + '_n' + str(num_trial + 1) + '.csv'
    cluster_df = pandas.read_csv(os.path.join(sys.path[0], "KMeans_Data/" + file_string))
    cluster_df = cluster_df.drop(cluster_df.columns[0], axis=1)
    cluster_df = cluster_df.transpose()
    cluster_list = cluster_df.values.tolist()

    set_list = [None] * k
    c = 0

    for cluster in cluster_list:
        cluster = [cell for cell in cluster if not numpy.isnan(cell)]
        set_list[c] = set(cluster)
        c += 1

    return set_list


def analyzeFeatures(df, k, numTrial):
    df = df.drop(['Name'], axis=1)
    spread_list = []

    cluster_list = createClusterList(df, k, numTrial)
    center = calculateCenter(df, cluster_list, k)
    cluster_spreads = []

    # for each cluster
    for cluster_id in range(k):
        n = len(cluster_list[cluster_id])

        # for each point in each cluster
        for cell_id in cluster_list[cluster_id]:

            feature_spread = [0] * 20

            # for each feature of each cell
            for x in range(20):
                feature_spread[x] += abs(df.loc[cell_id][x].astype('float64') - center[cluster_id][x])

        for x in range(20):
            feature_spread[x] = float("{0:.2f}".format(feature_spread[x] / n))

        cluster_spreads.append(feature_spread)

    cluster_features = []

    for x in cluster_spreads:
        top_features = []

        for y in range(5):
            index = x.index(min(x))
            top_features.append(index)
            x[index] = 100

        cluster_features.append(top_features)

    return cluster_features


def calculateCenter(df, set_list, k):
    n = len(set_list)
    dict_of_points = {}

    for x in range(n):
        running_total = {}
        dimensions = []
        len_list = len(set_list[x])
        for y in range(len_list):
            for z in df.columns:
                if z not in running_total:
                    running_total.update({z: df.loc[y, z].astype('float64')})
                else:
                    running_total.update({z: running_total.get(z) + df.loc[y, z].astype('float64')})
        for q in df.columns:
            if running_total.get(q) is None:
                running_total.update({q: 0})

            if len_list == 0:
                dimensions.append(0)
            else:
                dimensions.append(running_total.get(q) / len_list)
        dict_of_points.update({x: dimensions})
    centers = pandas.DataFrame(data=dict_of_points)
    return centers


def calculateAverageSpread(df, k, numTrials):
    df = df.drop(['Name'], axis=1)
    spread_list = []

    # for each trial
    for trial_id in range(numTrials):
        trial_list = []
        cluster_list = createClusterList(df, k, trial_id)
        centers = calculateCenter(df, cluster_list, k)

        # for each cluster
        for cluster_id in range(k):
            n = len(cluster_list[cluster_id])
            cluster_spread = 0

            # for each point in each cluster
            for cell_id in cluster_list[cluster_id]:
                cell_spread = 0

                # for each feature of each cell
                for x in range(20):
                    cell_spread += spatial.distance.euclidean(df.loc[cell_id].astype('float64'), centers[cluster_id])

                cluster_spread += cell_spread

            trial_list.append(cluster_spread / n)

        spread_list.append(sum(trial_list) / k)

    return spread_list
    # total_spread = 0
    # for x in spread_list:
    #     for y in x:
    #         total_spread += y
    # return total_spread/ (numTrials * k)

    # cluster_total_spread = {}
    #
    # for y in range(numTrials):
    #     for z in range(k):
    #         if z in cluster_total_spread:
    #             #previous = cluster_total_spread.get(z)
    #             cluster_total_spread.update({z: cluster_total_spread.get(z) + spread_list[y][z]})
    #         else:
    #             cluster_total_spread.update({z: spread_list[y][z]})
    #
    # for q in range(k):
    #     cluster_total_spread.update({q: cluster_total_spread.get(q)/numTrials})
    #
    # return cluster_total_spread


def visualizeData(data_list, numTrials):
    average_list = []
    total_list = [item for sublist in data_list for item in sublist]
    x_axis = []
    n = len(data_list)

    for x in range(numTrials):
        average_list.append(sum(data_list[x]) / numTrials)
        for y in range(numTrials):
            x_axis.append(3 + x)

    print("Total List")
    print(total_list)
    print("Average List")
    print(average_list)

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
    print(analyzeFeatures(df, 5, 1))
    # trial_average_spreads = []
    # for x in range(3,8):
    #     trial_average_spreads.append(calculateAverageSpread(df, x, numTrials))
    # print(trial_average_spreads)
    # visualizeData(trial_average_spreads, numTrials)

    print("FINISHED")
