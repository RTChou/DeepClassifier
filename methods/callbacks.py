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


class SimilarityCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.sim = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        self.aucs.append(roc_auc_score(self.validation_data[1][0], y_pred[0]))
        return





    def get_similarity(valid_smp, trainX, train_smp, txt_labels, top):
        for i in range(len(valid_smp)):
            sample = train_smp[valid_smp[i].item()] # sample name
            label = txt_labels[valid_smp[i].item()] # label
            
            sim = []
            for j in range(trainX.shape[0]):
                out = validation_model.predict_on_batch([trainX[valid_smp[i].item()], trainX[j]])
                sim.append(out)
    
    
    def get_sim(smp_ind, trainX):
        sim = []
        for i in range(trainX.shape[0]):
    
            out = validation_model.predict_on_batch()
            sim.append(out)
        return sim

