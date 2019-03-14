import numpy as np

def similarity_callback(smp_val, dat, val_model, top=10, random_seed=308):
    """
        smp_val: validatation datasets from split_data()
            dat: original input data, including sample names, expression data, and labels
            top: number of nearest samples
    random_seed: seed for random number generator
    """
    log_str = ''
    for i in range(valid_size):
        sim = []
        target = smp_val['inp'][i]
        for j in range(len(dat['smp'])):
            context = dat['inp'][j]
            out = val_model.predict_on_batch([target, context])
            sim.append(out[1])

        nearest = (-sim).argsort()[1:top + 1]
        smp_name = smp_val['smp'][i]
        label = smp_val['out'][i]
        nst_smp_names = dat['smp'][nearest]
        nst_labels = dat['out'][nearest]
        log_str = 'Nearest to %s (%s): ' % (smp_name, label)
        
        for k in range(top):
            log_str = log_str + '%s (%s)' % (nst_smp_names[k], nst_labels[k])
        
        log_str = log_str + '\n'

    return log_str

