import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from negative_sampling import sample_context_dist

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
    
    data = np.array(data)
    labels = pd.DataFrame(labels)

    return data, labels

def graph_embed(data, nb_neighbors=2):
    flat_list = []
    for i in range(data.shape[0]):
        sample = []
        for j in range(data.shape[1]):
            sample.append(data[i,j].item())
        flat_list.append(sample)

    nbrs = NearestNeighbors(nb_neighbors, algorithm='ball_tree').fit(flat_list)
    graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()

    return graph

def sample_training_set(sample_size, random_seed=123):
    np.random.seed(random_seed)
    input1_ind = []
    input2_ind = []
    output2 = []
    for i in range(sample_size):
        sample = sample_context_dist(graph, txt_labels, 0.5, 0.5, 20, 2)
        input1_ind.append(sample[0])
        input2_ind.append(sample[1])
        output2.append(sample[3])

