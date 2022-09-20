import graphity
import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
from sys import argv
import time

from interruptingcow import timeout
from func_timeout import func_set_timeout
import func_timeout
# def read_label():
#     # read label file
#     label_dict = {'BenignWare':0, 'Mirai':1, 'Tsunami':2, 'Hajime':3, 'Dofloo':4, 'Bashlite':5, 'Xorddos':6, 'Android':7, 'Pnscan':8, 'Unknown':9}
#     label = {}
#     threshold = {}
#     with open('/home/connlab/ChiaYi/CFGtest/CFG/dataset.csv', newline='') as csvfile:
#         rows = csv.reader(csvfile)
#         next(rows)
#         for row in rows:
#             threshold[row[0]] = row[2]
#             label[row[0]] = label_dict[row[1]]
#     print('---- finish read label ----\n')
#     return label



def Extract_feature(path):
    feature = []
    # Create graph
    G = graphity.gpickle2graph(path)

    # append #nodes & # edges
    feature.append(G.number_of_nodes())
    feature.append(G.number_of_edges())

    # append Density
    feature.append(graphity.get_density(G))

    # append Closeness Centrality
    for i in graphity.closeness_centrality(G):
        feature.append(i)

    # append Betweeness Centrality
    for i in graphity.betweeness_centrality(G):
        feature.append(i)

    # append Degree Centrality
    for i in graphity.degree_centrality(G):
        feature.append(i)

    # append Shortest Path
    for i in graphity.shortest_path(G):
        feature.append(i)

    return feature


def WriteCSV(text, csvPath):
    with open(csvPath, 'a+', newline='') as data:
        writer = csv.writer(data)
        writer.writerow(text)

def normalizePath(path):
    '''
    param. path
    description.
        back slash to slash
        delete the last char if it is slash
    '''
    path = path.replace('\\','/')
    if path[-1] == '/':
        path = path[:-1]
    return path  

def main(Path, OutputDir):
    '''
    param.
        Path: the dir where the *.gickle are in 
        OutputDir: the dir where benign.csv, malware.csv, record.csv are in 
    description.
        1. extract the feature of all the .gickle in [Path]
        2. save its feature, record to .csv in [OutputDir]
    '''

    # setting the path
    Path,OutputDir = normalizePath(Path), normalizePath(OutputDir)
    print('Input dir: ',Path)
    print('Output dir: ',OutputDir)
    featureCsvPath = OutputDir + '/' + Path.split('/')[-1] + '_Feature.csv'
    recordCsvPath = OutputDir + '/' + Path.split('/')[-1] + '_Record.csv'
    print('Record CSV: ',recordCsvPath)
    print('Feature CSV: ',featureCsvPath)

    # visit all the bin file under the root dir, transform to feature vector
    for root, dirs, files in os.walk(Path):
        if len(files) == 0:
            # means no file in the root
            continue

        files.sort()

        for index, f in enumerate(files):
            print(index, f)

            fpath = root + '/' + f
            name = f.split('.')[0]

            feature = [name]
            record = [name]

            # feature extraction & record
            try:
                # gpickle to feature vector
                start_time = time.time()
                feature += Extract_feature(fpath)
                end_time = time.time()
                cost = end_time - start_time
                # write feature vector to CSV
                WriteCSV(feature, featureCsvPath)
                # write record to csv
                record.append(cost)
                WriteCSV(record,recordCsvPath)
            except:
                # write record to csv
                record.append(-1)
                WriteCSV(record,recordCsvPath)
            


if __name__ == '__main__':
    Path = argv[1]
    OutputDir = argv[2]
    main(Path, OutputDir)
