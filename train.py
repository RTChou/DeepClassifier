# import matplotlib
# matplotlib.use('Agg')

import pandas as pd
import numpy as np
import argparse
from methods.utils import load_data, graph_embed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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
    parser.add_argument('-b', '--label_bin', required=True, help='path to output label binarizer')
    parser.add_argement('-p', '--plot', required=True, help='path to output accuracy/loss plot')
    # parser.add_argement('-t', '--epochs', default=75, type=int, help='number of epochs to train for')
    args = parser.parse_args()

    # params
    exp_path = args.exp
    label_path = args.label
    model_path = args.model
    label_bin_path = args.label_bin
    plot_path = args.plot
    
    nb_neighbors = 2
    sample_size = 10000

    # load data, shuffle the samples, and scale data
    print('[INFO] loading training data...')
    data, labels = load_data(exp_path, label_path)

    # convert training data to graph embedding
    print('[INFO] creating KNN graph from training data...')
    graph = graph_embed(data, nb_neighbors)

    # sample context distribution for training
    print('[INFO] sampling from graph and label context...')
    input1_ind, input2_ind, output2 = sample_training_set(sample_size, graph, labels)
    
    
    
    
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

    # sample validation examples
    valid_size = 5
    valid_smp = np.random.randint(trainX.shape[0], size=valid_size)
 
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

