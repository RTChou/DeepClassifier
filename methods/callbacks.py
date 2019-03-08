import keras
from sklearn.metrics import roc_auc_score
import numpy as np

class HistoryCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        self.aucs.append(roc_auc_score(self.validation_data[1][0], y_pred[0]))
        return



def similarity_callback(valid_inp, valid_out, inp_smp, out_smp, valid_size=10, top=10, random_seed=38):
    np.random.seed(random_seed)
    ind = np.random.randint(valid_inp.shape[0], size=valid_size)
    for i in range(valid_size):
        sample = valid_inp[ind[i]] # sample name
        label =  # label
        
        sim = []
        for j in range(trainX.shape[0]):
            out = validation_model.predict_on_batch([trainX[valid_smp[i].item()], trainX[j]])
            sim.append(out)

