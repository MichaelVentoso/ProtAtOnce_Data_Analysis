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

    set_list = [None]*k
    c = 0

    for cluster in cluster_list:
        cluster = [cell for cell in cluster if not numpy.isnan(cell)]
        set_list[c] = set(cluster)
        c += 1

    return set_list


def calculateCenter(df, set_list, k):
    n = len(set_list)
    dict_of_points = {}
    df = df.drop(['Name'], axis=1)

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
    total_spread = 0

    for i in range(numTrials):
        cluster_list = createClusterList(df, k, i)
        centers = calculateCenter(df, cluster_list, k)

        


if __name__ == '__main__':
    print("STARTED")
    print()

    df = createCellDataFrame()
    calculateAverageSpread(df, 3, 5)


    print("FINISHED")
