import matplotlib
matplotlib.use('Agg')

import argparse
from methods.utils import load_data, graph_embed, sample_training_set, split_data, plot_loss_acc
from sklearn.preprocessing import LabelBinarizer
from methods.graphSemiCNN import GraphSemiCNN
from methods.similarityCallback import SimilarityCallback
from sklearn.metrics import classification_report
import pickle

def main():
    parser = argparse.ArgumentParser(description="This script is for training the semi-supervised neural network model")
    parser.add_argument('-e', '--exp', required=True, help='path to input gene expression data, with genes in rows and samples in columns')
    parser.add_argument('-l', '--label', required=True, help='path to input labels')
    parser.add_argument('-m', '--model', required=True, help='path to output trained model')
    parser.add_argument('-b', '--label_bin', required=True, help='path to output label binarizer')
    parser.add_argement('-p', '--plot', required=True, help='path to output accuracy/loss plot')
    parser.add_argement('-c', '--epochs', default=75, type=int, help='number of epochs to train for')
    args = parser.parse_args()

    # params
    exp_path = args.exp
    label_path = args.label
    model_path = args.model
    label_bin_path = args.label_bin
    plot_path = args.plot
    nb_epochs = args.epochs
    
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
    print('[INFO] splitting data into training, validation, and testing sets...')
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
    nb_genes = inp['train'][0].shape[1]
    model = GraphSemiCNN.build(nb_genes, nb_classes)
    similarities = SimilarityCallback()
    fit_history = model.fit(inp['train'], out['train'], validation_data=(inp['validate'], out['validate']), 
            epochs=nb_epochs, batch_size=batch_size, callbacks=[similarities]) 

    # evaluate the model
    print('[INFO] evaluating the model...')
    predictions = model.predict(inp['test'], batch_size=batch_size)
    print(classification_report(out['test'][0].argmax(axis=1),
	predictions[0].argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy
    plot_loss_acc(plot_path, nb_epochs, fit_history)

    # save the model and label binarizer
    print('[INFO] serializing the model and label binarizer...')
    model.save(model_path)
    f = open(label_bin_path, 'wb')
    f.write(pickle.dumps(lb))
    f.close()

if __name__ == "__main__":
    main()

