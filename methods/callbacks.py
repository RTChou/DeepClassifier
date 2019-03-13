from sklearn.metrics import precision_recall_curve
import numpy as np

def history_callback(model):
    aucs = []
    y_pred = model.predict(self.validation_data[0])
    aucs.append(roc_auc_score(self.validation_data[1][0], y_pred[0]))
    return


def similarity_callback(val, dat, val_model, valid_size=10, top=10, random_seed=308):
    """
            val: validatation datasets from split_data()
            dat: original input data, including sample names, expression data, and labels
     valid_size: sample size for similarity validation
            top: number of nearest samples
    random_seed: seed for random number generator
    """
    np.random.seed(random_seed)
    ind = np.random.randint(len(val['smp']), size=valid_size)
    val = {'smp': val['smp'][0][ind], 'inp': val['inp'][0][ind], 'out': val['out'][0][ind]}
    for i in range(valid_size):
        sim = []
        target = val['inp'][i]
        for j in range(len(dat['smp'])):
            context = dat['inp'][j]
            out = val_model.predict_on_batch([target, context])
            sim.append(out[1])

        nearest = (-sim).argsort()[1:top + 1]
        smp_name = val['smp'][i]
        label = val['out'][i]
        nst_smp_names = dat['smp'][nearest]
        nst_labels = dat['out'][nearest]
        log_str = 'Nearest to %s (%s): ' % (smp_name, label)
        
        for k in range(top):
            log_str = log_str + '%s (%s)' % (nst_smp_names[k], nst_labels[k])
        
        print(log_str)






