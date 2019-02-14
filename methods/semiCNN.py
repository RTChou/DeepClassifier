import numpy as np
import pandas as pd

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.layers.pooling as pool
from keras.layers import Input, Activation, Flatten, Dense, Concatenate
"""
Graph based semi-supervised learning with convolution neural network
"""
def GraphSemiCNN(trainX, trainY):

    input_row = trainX.shape[0] # genes
    input_col = trainX.shape[1] # samples

    filters = 1000 # N filters
    kernal_size = 10 # a window of size k
    L1CNN = 0
    # dropout = 0.75 # parm for preventing overfitting
    actfun = 'relu'
    pool_size = 11 # a window of p features
    units1 = 11000 # number of nodes in hidden layer
    units2 = 5500
    units3 = 5500
    units4 = 5500
    nb_classes = 10
    nb_nodes = input_row # number of input samples
    optimization='sgd' # stochastic gradient descent
    
    input = Input(shape=(input_row, input_col))
    feature = conv.Conv1D(filters, kernal_size, init='he_normal', W_regularizer= l1(L1CNN), border_mode='same')(input)
    # initializer, regularizer, other params for conv
    # feature = Dropout(dropout)(feature)
    feature = Activation(actfun)(feature)
    feature = pool.MaxPooling1D(pool_size)(feature)
    feature = Flatten()(feature)
    
    hidden1 = Dense(units1, activation='relu')(feature)
    hidden2 = Dense(units2, activation='relu')(hidden1) # z1 -> z2
    hidden3 = Dense(units3, activation='relu')(hidden1) # z1 -> z3
    hidden4 = Dense(units4, activation='relu')(hidden3) # z3 -> z4
    concatenated = Concatenate(axis=0)([hidden2, hidden4]) # concatenate z2, z4
    
    output1 = Dense(nb_classes, activation='softmax')(concatenated)
    output2 = Dense(nb_nodes, activation='softmax')(hidden3)

    cnn = Model(input, [output1, output2])
    cnn.compile(loss='categorical_crossentropy', optimizer=optimization, metrics=['accuracy']) # configure before training
    # loss function: cross entropy

