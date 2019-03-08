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



def similarity_callback(val, dat, val_model, valid_size=10, top=10, random_seed=38):
    """
            val: validatation datasets, indluding sample names, expression data, and labels
            dat: sampled datasets from original input data, including sample names, expression data, and labels
     valid_size: sample size for similarity validation
            top: number of nearest samples
    random_seed: seed for random number generator
    """
    np.random.seed(random_seed)
    ind = np.random.randint(len(val['smp']), size=valid_size)
    for i in range(valid_size):
        sim = []
        target = val['inp'][ind[i]]
        for j in range(len(dat['smp'])):
            context = dat['inp'][j]
            out = val_model.predict_on_batch([target, context])
            sim.append(out[1])

        smp_name = val['smp'][ind[i]]
        label = val['out'][ind[i]]
        nearest = (-sim).argsort()[1:top + 1]
        nst_smp_names = dat['smp'][nearest]
        nst_labels = dat['out'][nearest]
        log_str = 'Nearest to %s (%s): ' % (smp_name, label)
        
        for k in range(top):
            log_str = log_str + '%s (%s)' % (nst_smp_names[k], nst_labels[k])
        
        print(log_str)






