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

# sample context dist
from methods.negative_sampling import sample_context_dist
np.random.seed(123)
for i in range(10):
    sample_context_dist(graph, txt_labels, 0.5, 0.5, 3, 2)

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
    plt.text(x, y - 0.08, style='italic', fontweight='bold', s=txt_labels[i].item() + '\n(' + train_smp[i].item() + ')', horizontalalignment='center')

fig.savefig('graph.png', transparent=True)
plt.show()

# ---------------

filters = 1000
kernel_size = 10
L1CNN = 0
# dropout = 0.75 # parm for preventing overfitting
actfun = 'relu'
pool_size = 11 # window of eatures
units1 = 11000 # numberof nodes in hidden layer
units2 = 5500
units3 = 5500
units4 = 5500
INIT_LR = 0.01 # initial learning rate
# EPOCHS = 75
lamd = 1.0

input1 = Input(shape=(input_genes, 1))
feature1 = conv.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l1(L1CNN))(input1)
feature1 = Activation(actfun)(feature1)
feature1 = pool.MaxPooling1D(pool_size)(feature1)
feature1 = Flatten()(feature1)

input2 = Input(shape=(input_genes, 1))
feature2 = conv.Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l1(L1CNN))(input2)
feature2 = Activation(actfun)(feature2)
feature2 = pool.MaxPooling1D(pool_size)(feature2)
feature2 = Flatten()(feature2)

hidden1_1 = Dense(units1, activation='relu')(feature1)
target = Dense(units3, activation='relu')(hidden1_1) # z1 -> z3
hidden1_2 = Dense(units1, activation='relu')(feature2)
context = Dense(units3, activation='relu')(hidden1_2)
hidden2 = Dense(units2, activation='relu')(hidden1_1) # z1 -> z2
hidden4 = Dense(units4, activation='relu')(target) # z3 -> z4
concatenated = Concatenate(axis=0)([hidden2, hidden4]) # concatenate z2, z4

dot_product = Dot(axes=1)([target, context])

