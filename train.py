# import matplotlib
# matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
from methods.semiCNN import build_model

# from sklearn.metrics import classification_report
# from imutils import paths
# import matplotlib.pyplot as plt
# import pickle
# import cv2 # not being used
# import os
# import keras.models as models

def main():
    parser = argparse.ArgumentParser(description="This script is for training the semi-supervised neural network model")
    parser.add_argument('-e', '--exp', required=True, help='path to input gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-l', '--label', required=True, help='path to input labels')
    parser.add_argument('-m', '--model', required=True, help='path to output trained model')
    parser.add_argument('-b', '--label-bin', required=True, help='path to output label binarizer')
    parser.add_argement('-p', '--plot', required=True, help='path to output accuracy/loss plot')
    args = parser.parse_args()

    print('[INFO] loading training data...')
    data = [] # training data
    labels = []
    
    # load data, shuffle the samples, and scale data
    exp_dst = pd.read_csv(args.exp, sep='\t', index_col=0)
    label_dst = pd.read_csv(args.label, sep='\t', index_col=0)
    
    exp_dst = exp_dst.sample(frac=1, random_state=33, axis=1)
    exp_dst = pd.DataFrame(scale(exp_dst), index=exp_dst.index, columns=exp_dst.columns)
    label_dst = label_dst.replace(np.nan, '', regex=True)

    # store data in list
    for i in range(exp_dst.shape[1]):
       exp = list(exp_dst.iloc[:,i].values)
       label = label_dst.loc[[exp_dst.columns[i]]]['tissue'].item()
       data.append([[i] for i in exp])
       labels.append([label])
    
    data = np.array(data)
    labels = np.array(labels)

    # split the data into training and test sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=33)
    
    # convert trainX to graph embedding
    flat_list = []
    for i in range(trainX.shape[0]):
        sample = []
        for j in range(trainX.shape[1]):
            sample.append(trainX[i,j].item())
        flat_list.append(sample)
    
    nbrs = NearestNeighbors(n_neighbors=1000, algorithm='ball_tree').fit(flat_list)
    graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()
    labels_t = trainY

    # one-hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    nb_classes = len(lb.classes_)

    # initialize the model
    models = build_model(trainX, trainY, testX, testY, nb_classes)

if __name__ == "__main__":
    main()

