import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from .negativeSampling import NegativeSampling
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
    
    return {'smp': samples, 'inp': data, 'out': labels}


def sample_training_set(dat, sample_size, nb_neighbors=2, random_seed1=123, r1=0.5, r2=0.5, q=100, d=10, portion=[.6, .2], random_seed2=33):
    """
             dat: output from load_data
     sample_size: number of representative samples
    nb_neighbors: number of neighbors for calcularing k-nearest neighbors
    random_seed1: seed for sampling from context
         poriton: portion of training, validatation, and test sets
    random_seed2: seed for splitting the dataset
    """
    # construct graph
    print('[INFO] creating KNN graph from data...')
    flat_list = []
    for i in range(dat['inp'].shape[0]):
        sample = []
        for j in range(dat['inp'].shape[1]):
            sample.append(dat['inp'][i,j].item())
        flat_list.append(sample)
    nbrs = NearestNeighbors(nb_neighbors, algorithm='ball_tree').fit(flat_list)
    graph = nbrs.kneighbors_graph(flat_list, mode='distance').toarray()

    # sample context dist
    print('[INFO] sampling from graph and label context...')
    np.random.seed(random_seed1)
    input1_ind = []
    input2_ind = []
    output2 = []
    ns = NegativeSampling()
    pair_sets = ns.get_label_pairs(dat['out'])
    
    for i in range(sample_size):
        sample = ns.sample_context_dist(graph, dat['out'], r1, r2, q, d, pair_sets)
        input1_ind.append(sample[0])
        input2_ind.append(sample[1])
        output2.append(sample[2])

    # split data into training, validation, and test sets
    print('[INFO] splitting data into training, validation, and testing sets...')
    trn, val, tst = split_data([dat['smp'][input1_ind], dat['smp'][input2_ind]], 
            [dat['inp'][input1_ind], dat['inp'][input2_ind]], 
            [dat['out'][input1_ind], np.array(output2)], 
            outputs, poriton, random_seed2)    
    
    return trn, val, tst


def split_data(smp_names, inputs, outputs, portion=[.6, .2], random_seed=33):
    sample_size = inputs[0].shape[0]
    np.random.seed(random_seed)
    ind = np.arange(sample_size)
    np.random.shuffle(ind)
    train, valid, test = np.split(ind, [int(portion[0]*sample_size), int(sum(portion)*sample_size)])

    trn = {}
    trn['smp'] = [smp_names[0][train], smp_names[1][train]]
    trn['inp'] = [inputs[0][train], inputs[1][train]]
    trn['out'] = [outputs[0][train], outputs[1][train]]

    val = {}
    val['smp'] = [smp_names[0][valid], smp_names[1][valid]]
    val['inp'] = [inputs[0][valid], inputs[1][valid]]
    val['out'] = [outputs[0][valid], outputs[1][valid]]

    tst = {}
    tst['smp'] = [smp_names[0][test], smp_names[1][test]]
    tst['inp'] = [inputs[0][test], inputs[1][test]]
    tst['out'] = [outputs[0][test], outputs[1][test]]

    return trn, val, tst


def plot_loss_acc(plot_path, nb_epochs, fit_history):
    N = np.arange(0, nb_epochs)
    H = fit_history
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(N, H.history['loss'], label='train_loss')
    plt.plot(N, H.history['val_loss'], label='val_loss')
    plt.plot(N, H.history['acc'], label='train_acc')
    plt.plot(N, H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy (Semi-supervised NN)')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(plot_path)


