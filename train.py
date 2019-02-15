# import matplotlib
# matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import SGD
# from imutils import paths
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random
# import pickle
# import cv2 # not being used
# import os

def main():
    parser = argparse.ArgumentParser(description="This script is for training the semi-supervised neural network model")
    parser.add_argument('-e', '--exp', required=True, help='path to input gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-l', '--label', required=True, help='path to input labels')
    parser.add_argument('-m', '--model', required=True, help='path to output trained model')
    parser.add_argument('-b', '--label-bin', required=True, help='path to output label binarizer')
    parser.add_argement('-p', '--plot', required=True, help='path to output accuracy/loss plot')
    args = parser.parse_args()

    print('[INFO] loading training data...')
    data = []
    labels = []

    # load data and shuffle the samples
    exp_dst = pd.read_csv(args.exp, sep='\t', index_col=0)
    label_dst = pd.read_csv(args.label, sep='\t', index_col=0)
    exp_dst = exp_dst.sample(frac=1, random_state=33, axis=1)

    # store data in list
    for i in range(0, exp_dst.shape[1]):
       data.append(exp_dst.iloc[:,i].values)
       labels.append(label_dst.loc[[exp_dst.columns[i]]]['tissue'].item())
    
    # scale data
    data = scale(np.array(data, dtype='float'), axis=1)
    labels = np.array(labels)
    
    # split the data into training and test sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=33)

    # one-hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    nb_classes = len(lb.classes_)



    models=GraphSemiCNN()

if __name__ == "__main__":
    main()

