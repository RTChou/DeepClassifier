import matplotlib
matplotlib.use('Agg')

import argparse
from methods.utils import load_data, graph_embed, sample_training_set, split_data, plot_loss_acc
from sklearn.preprocessing import LabelBinarizer
from methods.graphSemiCNN import GraphSemiCNN
from methods.similarityCallback import HistoryCallback, SimilarityCallback
from sklearn.metrics import classification_report
import pickle

exp_path = '~/Downloads/imputation/rnaseq_data_tpm_from_metadata.tsv'
label_path = '~/Downloads/imputation/rnaseq_label_from_metadata.tsv'
# model_path = args.model
# label_bin_path = args.label_bin
# plot_path = args.plot

nb_epochs = 75
nb_neighbors = 2
sample_size = 10000
batch_size = 32

# load data, shuffle the samples, and scale data
print('[INFO] loading training data...')
samples, data, labels = load_data(exp_path, label_path)

# convert data to graph embedding
print('[INFO] creating KNN graph from data...')
graph = graph_embed(data, nb_neighbors)

# sample context distribution
print('[INFO] sampling from graph and label context...')
input1_ind, input2_ind, output2 = sample_training_set(sample_size, graph, labels)
smp_names = [samples[input1_ind], samples[input2_ind]]
inputs = [data[input1_ind], data[input2_ind]]
outputs = [labels[input1_ind], output2]

# 60% train set, 20% validation set, 20% test set 
smp, inp, out = split_data(smp_names, inputs, outputs, portion=[.6, .2])

# one-hot encoding
lb = LabelBinarizer()
lb.fit(labels)
out['train'][0] = lb.transform(out['train'][0])
out['validate'][0] = lb.transform(out['validate'][0])
out['test'][0] = lb.transform(out['test'][0])
nb_classes = len(lb.classes_)

# build and train the model
print('[INFO] building and training the model...')
nb_samples = inp['train'][0].shape[0]
nb_genes = inp['train'][0].shape[1]
model = GraphSemiCNN.build(nb_genes, nb_classes)




np.savetxt('train_smp.csv',train_smp,delimiter=',',fmt="%s")
np.savetxt('txt_labels.csv',txt_labels,delimiter=',',fmt="%s")
np.savetxt('graph.csv',graph,delimiter=',')

# sample context dist
from methods.negative_sampling import sample_context_dist
np.random.seed(123)
for i in range(10):
    sample_context_dist(graph, txt_labels, 0.5, 0.5, 3, 2)

# plot graph (in python 2.7)
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
dot_product = Reshape((1,))(dot_product)

output1 = Dense(nb_classes, activation='softmax', name='output1')(concatenated)
output2 = Dense(1, activation='sigmoid', name='output2')(dot_product) # softmax?

model = Model(inputs=[input1, input2], outputs=[output1, output2], name='graphSemiCNN')

losses = {
    'output1': 'categorical_crossentropy',
    'output2': 'binary_crossentropy',
}
lossWeights = {'output1': 1.0, 'output2': lambd}
opt = SGD(lr=INIT_LR)

model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['accuracy']) # loss function: cross entropy

