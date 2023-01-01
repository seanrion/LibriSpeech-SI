from test import eval_model
from Constant import NUM_FRAMES,NUM_FBANKS,CHECKPOINTS_SOFTMAX_DIR,WORKING_DIR,CHECKPOINTS_TRIPLET_DIR
from conv_models import DeepSpeakerModel
from natsort import natsorted
from glob import glob
import os
import csv

if __name__ == '__main__':
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    header = ['Epochs','f-measure','true_positive_rate','accuracy','equal_error_rate']
    dsm = DeepSpeakerModel(batch_input_shape)


    rows = []
    checkpoints = natsorted(glob(os.path.join(CHECKPOINTS_SOFTMAX_DIR, '*.h5')))
    Epochs = []
    for name in checkpoints:
        Epochs.append(name.removesuffix('.h5').split('_')[-1])
    for i in range(len(checkpoints)):
        dsm.m.load_weights(checkpoints[i], by_name=True)
        fm, tpr, acc, eer = eval_model(os.path.join(WORKING_DIR,'train'), model=dsm)
        rows.append([Epochs[i], fm, tpr, acc, eer])
    with open('pretrain_models_eval.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    rows = []
    checkpoints = natsorted(glob(os.path.join(CHECKPOINTS_TRIPLET_DIR, '*.h5')))
    Epochs = []
    for name in checkpoints:
        Epochs.append(name.removesuffix('.h5').split('_')[-1])
    for i in range(len(checkpoints)):
        dsm.m.load_weights(checkpoints[i], by_name=True)
        fm, tpr, acc, eer = eval_model(os.path.join(WORKING_DIR,'train'), model=dsm)
        rows.append([Epochs[i], fm, tpr, acc, eer])
    with open('train_models_eval.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)