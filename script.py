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
label_dst = label_dst.replace(np.nan, 'Unlabeled', regex=True)

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
trainX = trainX[0:10]
trainY = trainY[0:10]
flat_list = []
for i in range(trainX.shape[0]):
    sample = []
    for j in range(trainX.shape[1]):
        sample.append(trainX[i,j].item())
    flat_list.append(sample)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(flat_list)
graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()
labels_t = trainY


# plot graph
from numpy import genfromtxt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

graph = genfromtxt('graph.csv', delimiter=',')
df=pd.read_csv('labels_t.csv', sep=',',header=None)
labels_t = df.values

G=nx.Graph()
pos = nx.spring_layout(G)

node_labels = {}
edge_labels = {}
for i in range(10):
    G.add_node(i)
    node_labels[i] = labels_t[i].item()

for i in range(10):
    for j in range(10):
        G.add_edge(i,j,weight=graph[i][j])
        edge_labels[(i,j)] = graph[i][j]

plt.figure()
G=nx.relabel_nodes(G, node_labels, copy=False)
nx.draw(G, pos, with_labels=True, node_size=800, node_color='blue', edge_color='grey', alpha=0.9)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
plt.axis('off')
# plt.savefig("labels_and_colors.png") # save as png
plt.show() # display

