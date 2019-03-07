import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from .negative_sampling import sample_context_dist, get_label_pairs
import matplotlib.pyplot as plt

def load_data(exp_path, label_path, random_state=33):
    data = [] # training data
    labels = [] 
    
    # load data, shuffle the samples, and scale data
    exp_dst = pd.read_csv(exp_path, sep='\t', index_col=0)
    label_dst = pd.read_csv(label_path, sep='\t', index_col=0)    
    exp_dst = exp_dst.sample(frac=1, random_state=random_state, axis=1)
    exp_dst = pd.DataFrame(scale(exp_dst), index=exp_dst.index, columns=exp_dst.columns)
    label_dst = label_dst.replace(np.nan, 'unlabeled', regex=True)    
    
    # store data in list
    for i in range(exp_dst.shape[1]):
       exp = list(exp_dst.iloc[:,i].values)
       label = label_dst.loc[[exp_dst.columns[i]]]['tissue'].item()
       data.append([[i] for i in exp])
       labels.append([label])    
   
    samples = exp_dst.columns
    data = np.array(data)
    labels = np.array(labels)
    
    return samples, data, labels


def graph_embed(data, nb_neighbors=2):
    """
            data: training data in array format
    nb_neighbors: number of neighbors for calcularing k-nearest neighbors
    """
    flat_list = []
    for i in range(data.shape[0]):
        sample = []
        for j in range(data.shape[1]):
            sample.append(data[i,j].item())
        flat_list.append(sample)
    nbrs = NearestNeighbors(nb_neighbors, algorithm='ball_tree').fit(flat_list)
    graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()
    return graph


def sample_training_set(sample_size, graph, labels, random_seed=123, r1=0.5, r2=0.5, q=100, d=10):
    np.random.seed(random_seed)
    input1_ind = []
    input2_ind = []
    output2 = []
    pair_sets = get_label_pairs(labels)
    
    for i in range(sample_size):
        sample = sample_context_dist(graph, labels, r1, r2, q, d, pair_sets)
        input1_ind.append(sample[0])
        input2_ind.append(sample[1])
        output2.append(sample[2])
    return input1_ind, input2_ind, output2


def split_data(smp_names, inputs, outputs, portion=[.6, .2], random_seed=33):
    sample_size = inputs[0].shape[0]
    np.random.seed(random_seed)
    ind = np.arange(sample_size)
    np.random.shuffle(ind)
    train, validate, test = np.split(ind, [int(portion[0]*sample_size), int(sum(portion)*sample_size)])

    smp = {}
    smp['train'] = [smp_names[0][train], smp_names[1][train]]
    smp['validate'] = [smp_names[0][validate], smp_names[1][validate]]
    smp['test'] = [smp_names[0][test], smp_names[1][test]]

    inp = {}
    inp['train'] = [inputs[0][train], inputs[1][train]] 
    inp['validate'] = [inputs[0][validate], inputs[1][validate]]
    inp['test'] = [inputs[0][test], inputs[1][test]]
    
    out = {}
    out['train'] = [outputs[0][train], outputs[1][train]]
    out['validate'] = [outputs[0][validate], outputs[1][validate]]
    out['test'] = [outputs[0][test], outputs[1][test]]
    
    return smp, inp, out

def plot_loss_acc(plot_path, nb_epochs, fit_history):
    N = np.arrange(0, nb_epochs)
    H = fit_history
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(N, H.history['loss'], label='train_loss')
    plt.plot(N, H.history['val_loss'], label='val_loss')
    plt.plot(N, H.history['acc'], label='train_acc')
    plt.plot(N, H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(plot_path)

