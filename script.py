import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
from methods.graphSemiCNN import build_model

print('[INFO] loading training data...')
data = [] # training data
labels = []

# load data, shuffle the samples, and scale data
exp_dst = pd.read_csv('~/Downloads/imputation/rnaseq_data_tpm_from_metadata.tsv', sep='\t', index_col=0)
label_dst = pd.read_csv('~/Downloads/imputation/rnaseq_label_from_metadata.tsv', sep='\t', index_col=0)

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
trainX = trainX[0:50]
trainY = trainY[0:50]
flat_list = []
for i in range(trainX.shape[0]):
    sample = []
    for j in range(trainX.shape[1]):
        sample.append(trainX[i,j].item())
    flat_list.append(sample)

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(flat_list)
graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()
labels_t = trainY

