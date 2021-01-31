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


def createCellDataFrame():

    #creates dataframe for first .csv file and sets column names
    cell_dataset = pandas.read_csv(os.path.join(sys.path[0], 'cell_dataset.csv'))

    cell_dataframe = pandas.DataFrame(cell_dataset,
                                       columns=['Name', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
                                                'x_6', 'x_7', 'x_8', 'x_9', 'x_10',
                                                'x_11', 'x_12', 'x_13', 'x_14', 'x_15',
                                                'x_16', 'x_17', 'x_18', 'x_19', 'x_20'])

    return cell_dataframe


def calculateSpaceBounds(df):

    df = df.drop(['Name'], axis=1)

    bounds = []

    for label, content in df.iteritems():
        bounds.append([content.min(), content.max()])

    return bounds


def createKPoints(k,bounds):

    dict_of_points = {}

    for x in range(k):
        dimensions = []
        for y in range(20):
            dimensions.append(numpy.random.uniform(bounds[y][0], bounds[y][1]))
        dict_of_points.update({ x : dimensions })

    centers = pandas.DataFrame(data=dict_of_points)
    return centers


def findNearestCenter(df, centers, index):
    k = len(centers.columns)
    df = df.drop(['Name'], axis=1)

    min_dist = 1000000.0
    nearest = -1

    for x in range(k):
        dist = spatial.distance.euclidean(df.loc[index].astype('float64'), centers[x])
        if dist < min_dist:
            nearest = x
            min_dist = dist

    return nearest


def calculateCenters(df, set_list, k):
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
                running_total.update({q:0})

            if len_list == 0:
                dimensions.append(0)
            else:
                dimensions.append(running_total.get(q)/len_list)
        dict_of_points.update({x: dimensions})
    centers = pandas.DataFrame(data=dict_of_points)
    return centers


def kMeans(df,k,num,iterations,centers=None):

    bounds = calculateSpaceBounds(df)

    if centers is None:
        centers = createKPoints(k,bounds)

    num_points = len(df)

    set_list = []

    for i in range(iterations):
        print(i)
        set_list = [set() for s in range(k)]
        for x in range(num_points):
            nearest = findNearestCenter(df,centers,x)
            set_list[nearest].add(x)
        return
        centers = calculateCenters(df, set_list, k)

    df_return = pandas.DataFrame(list(set_list))
    df_return = df_return.transpose()
    df_return.to_csv('kMeans_k' + str(k) + '_n' + str(num) + '.csv')


if __name__ == '__main__':
    print("STARTED")
    print()

    cell_dataframe = createCellDataFrame();

    for x in range(1, 6):
        for y in range(3, 8, 1):
            kMeans(cell_dataframe, y, x, 50)

    print("FINISHED")