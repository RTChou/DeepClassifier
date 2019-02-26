import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
from methods.graphSemiCNN import build_model

data = [] # training data
labels = []

# load data, shuffle the samples, and scale data
exp_dst = pd.read_csv('~/Downloads/imputation/rnaseq_data_tpm_from_metadata.tsv', sep='\t', index_col=0)
label_dst = pd.read_csv('~/Downloads/imputation/rnaseq_label_from_metadata.tsv', sep='\t', index_col=0)

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

trainX = trainX[0:10]
train_smp = train_smp[0:10]
txt_labels = txt_labels[0:10]
# convert trainX to graph embedding
flat_list = []
for i in range(trainX.shape[0]):
    sample = []
    for j in range(trainX.shape[1]):
        sample.append(trainX[i,j].item())
    flat_list.append(sample)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(flat_list)
graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()

np.savetxt('train_smp.csv',train_smp,delimiter=',',fmt="%s")
np.savetxt('txt_labels.csv',txt_labels,delimiter=',',fmt="%s")
np.savetxt('graph.csv',graph,delimiter=',')

# plot graph
from numpy import genfromtxt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

df=pd.read_csv('train_smp.csv', sep=',',header=None)
train_smp = df.values
df=pd.read_csv('txt_labels.csv', sep=',',header=None)
txt_labels = df.values
graph = genfromtxt('graph.csv', delimiter=',')

fig = plt.figure(figsize=(8.0, 8.0))
seed = 43
np.random.seed(seed)
G=nx.Graph()
unlabled_smp=[]
for i in range(len(train_smp)):
    G.add_node(train_smp[i].item())
    if (txt_labels[i].item()) == 'unlabeled':
        unlabled_smp.append(train_smp[i].item())

for i in range(len(graph)):
    for j in range(len(graph)):
        if graph[i][j] != 0:
            G.add_edge(train_smp[i].item(), train_smp[j].item(), weight=round(graph[i][j], 2))

color_map = []
for node in G:
    if node in unlabled_smp:
        color_map.append('grey')
    else: 
        color_map.append('blue') 

labels = {}
for u,v,data in G.edges(data=True):
    labels[(u,v)] = data['weight']

pos=nx.random_layout(G)
nx.draw(G, pos, node_size=800, width=3, node_color=color_map, edge_color='grey')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
for i in range(len(train_smp)):
    x,y = pos[train_smp[i].item()]
    plt.text(x, y - 0.08, style='italic', s=txt_labels[i].item() + '\n(' + train_smp[i].item() + ')', horizontalalignment='center')

fig.savefig('graph.png', transparent=True)
plt.show()

