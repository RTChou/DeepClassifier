import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import cv2 # not being used
import os

def main():
    parser = argparse.ArgumentParser(description="This script is for training the semi-supervised neural network model")
    parser.add_argument('-i', '--input', required=True, help='path to input gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-l', '--label', required=True, help='path to input labels')
    parser.add_argument('-m', '--model', required=True, help='path to output trained model')
    parser.add_argument('-b', '--label-bin', required=True, help='path to output label binarizer')
    parser.add_argement('-p', '--plot', required=True, help='path to output accuracy/loss plot')
    args = parser.parse_args()

    print('[INFO] loading training data...')
    data = []
    labels = []

    trainX = pd.read_csv(args.input, sep='\t', index_col=0)
    trainY = pd.read_csv(args.label, sep='\t', index_col=0)
    trainX = trainX.sample(frac=1, random_state=33, axis=1)

    for i in range(0, trainX.shape[1]):
       data.append(trainX.iloc[:,i].values)
       labels.append(trainY.loc[[trainX.columns[i]]]['tissue'].item())
    
    data = scale(np.array(data, dtype='float'), axis=1)
    labels = np.array(labels)
    

    models=GraphSemiCNN()

if __name__ == "__main__":
    main()

