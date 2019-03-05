# import matplotlib
# matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
from methods.negative_sampling import sample_context_dist
from methods.graphSemiCNN import GraphSemiCNN
from sklearn.metrics import classification_report

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
    # parser.add_argement('-t', '--epochs', default=75, type=int, help='number of epochs to train for')
    args = parser.parse_args()

    print('[INFO] loading training data...')
    data = [] # training data
    labels = []
    
    # load data, shuffle the samples, and scale data
    exp_dst = pd.read_csv(args.exp, sep='\t', index_col=0)
    label_dst = pd.read_csv(args.label, sep='\t', index_col=0)
    
    exp_dst = exp_dst.sample(frac=1, random_state=33, axis=1)
    exp_dst = pd.DataFrame(scale(exp_dst), index=exp_dst.index, columns=exp_dst.columns)
    label_dst = label_dst.replace(np.nan, 'unlabeled', regex=True)

    # store data in list
    for i in range(exp_dst.shape[1]):
       exp = list(exp_dst.iloc[:,i].values)
       label = label_dst.loc[[exp_dst.columns[i]]]['tissue'].item()
       data.append([[i] for i in exp])
       labels.append([label])
    
    data = np.array(data)
    labels = pd.DataFrame(labels)

    # split the data into training and test sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=33)
    train_ind = trainY.index
    train_smp = exp_dst.columns[train_ind]   
    txt_labels = trainY.values # labels in txt format

    # one-hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    nb_classes = len(lb.classes_)

    print('[INFO] creating KNN graph from training data...')
    # convert trainX to graph embedding
    flat_list = []
    for i in range(trainX.shape[0]):
        sample = []
        for j in range(trainX.shape[1]):
            sample.append(trainX[i,j].item())
        flat_list.append(sample)
    
    nbrs = NearestNeighbors(n_neighbors=1000, algorithm='ball_tree').fit(flat_list)
    graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()

    print('[INFO] sampling from graph and label context...')
    np.random.seed(123)

    # sample context distribution for training
    sample_size = 2000
    input1_ind = []
    input2_ind = []
    output2 = []
    for i in range(sample_size):
        sample = sample_context_dist(graph, txt_labels, 0.5, 0.5, 20, 2)
        input1_ind.append(sample[0])
        input2_ind.append(sample[1])
        output2.append(sample[3])
    
    # sample validation examples
    valid_size = 5
    valid_samples = np.random.randint(trainX.shape[0], size=valid_size)
 
    print('[INFO] building and training the model...')
    # initialize the model
    model = GraphSemiCNN(trainX, trainY, testX, testY, nb_classes)
    
    # build and train the model
    model.build()
    model.train()

    # evaluate the model
    predictions = model.predict()
    print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

if __name__ == "__main__":
    main()

