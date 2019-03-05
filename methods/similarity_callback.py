import numpy as np

def run_sim(valid_smp, trainX, train_smp, txt_labels, top):
    for i in range(len(valid_smp)):
        sample = train_smp[valid_smp[i].item()] # sample name
        label = txt_labels[valid_smp[i].item()] # label


def get_sim(smp_ind, trainX):
    sim = []
    for i in range(trainX.shape[0]):

        out = validation_model.predict_on_batch()
        sim.append(out)
    return sim

