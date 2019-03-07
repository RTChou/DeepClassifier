import keras
import numpy as np

class SimilarityCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        
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

