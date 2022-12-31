import logging

import numpy as np
from tqdm import tqdm
import glob 
from audio import *
from batcher import *
from Constant import NUM_FBANKS, NUM_FRAMES, CHECKPOINTS_TRIPLET_DIR, BATCH_SIZE,CHECKPOINTS_SOFTMAX_DIR
from conv_models import DeepSpeakerModel
from eval_metrics import evaluate
from utils import load_best_checkpoint, enable_deterministic

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)

    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s


def inference_model(working_dir: str, model: DeepSpeakerModel):
    # load speaker in train
    train_audio=[]
    for i in tqdm(range(250)):
        train_audio_path=sorted(glob.glob('./audio_dir/train/spk'+str(i+1).zfill(3)+'/spk*_*[!noise].flac'))[0]
        # print(train_audio_path)
        mfcc_tmp=sample_from_mfcc(read_mfcc(train_audio_path, SAMPLE_RATE), NUM_FRAMES)
        train_audio.append(model.m.predict(np.expand_dims(mfcc_tmp, axis=0)))
    
    # load speaker in test
    test_audio_paths=sorted(glob.glob(os.path.join('./audio_dir/test/test*.flac')))
    scores_test=np.zeros(len(test_audio_paths))
    anchor_test=[]
    for i in tqdm(range(10) ,desc='predict'):
        test_audio_path=test_audio_paths[i]
        mfcc_tmp=sample_from_mfcc(read_mfcc(test_audio_path, SAMPLE_RATE), NUM_FRAMES)
        test_audio=model.m.predict(np.expand_dims(mfcc_tmp, axis=0))
        # test each utterence's similarity to each speaker in train
        for j in range(len(train_audio)):
            scores_test[i]=(batch_cosine_similarity(test_audio, train_audio[j]))
        # print("anchor speaker: ",np.argmax(scores_test))
        speaker_index=np.argmax(scores_test)+1
        print(speaker_index)
        anchor_test.append(test_audio_path.split('/')[-1]+' '+ 'spk' + str(speaker_index).zfill(3))
    return anchor_test


def inference(working_dir, checkpoint_file=None):
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    dsm = DeepSpeakerModel(batch_input_shape)
    if checkpoint_file is None:
        checkpoint_file = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
    if checkpoint_file is not None:
        logger.info(f'Found checkpoint [{checkpoint_file}]. Loading weights...')
        dsm.m.load_weights(checkpoint_file, by_name=True)
    else:
        logger.info(f'Could not find any checkpoint in {checkpoint_file}.')
        exit(1)

    anchor_test = inference_model(working_dir, model=dsm)
    anchor_test = [str(i) for i in anchor_test]  #转str
    anchor_test = '\r'.join(anchor_test)  #逗号分隔
    with open('2.txt', 'a', encoding='utf8') as f:  #写入
	    f.writelines(anchor_test+'\n')
import os
from Constant import WORKING_DIR
if __name__ == '__main__':
    # inference(os.path.join(WORKING_DIR,'train'),os.path.join(CHECKPOINTS_TRIPLET_DIR,'ResCNN_checkpoint_1.h5'))
    inference(os.path.join(WORKING_DIR,'test'),checkpoint_file=None)
