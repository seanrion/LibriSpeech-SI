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
import random
from collections import OrderedDict
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)

    s = np.sum(mul)
    
    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s


def inference_model(model: DeepSpeakerModel):
    # load speaker in train
    batcher = LazyTripletBatcher(os.path.join(WORKING_DIR,'train'), NUM_FRAMES, model)
    speakers = list(batcher.audio.speakers_to_utterances.keys())
    sp_to_utt = {**batcher.sp_to_utt_test, **batcher.sp_to_utt_train}
    spk_predictions = list()
    if not os.path.exists('embeddings.npz'):
        for speaker in tqdm(speakers):
            utterances = list(filter(lambda x :not 'noise' in x,sp_to_utt[speaker]))
            mfccs = np.vstack([
                [sample_from_mfcc_file(u, NUM_FRAMES) for u in utterances]
            ])
            predictions = model.m.predict(mfccs, batch_size=BATCH_SIZE)
            spk_predictions.append(predictions)
        all_params = OrderedDict([(speakers[i], spk_predictions[i]) for i in range(len(speakers))])
        np.savez('embeddings', **all_params)
    spk_predictions = np.load('embeddings.npz')

    # load speaker in test
    test_audio_paths=sorted(glob.glob(os.path.join('./audio_dir/test/test*.flac')))+sorted(glob.glob(os.path.join('./audio_dir/test-noisy/test*.flac')))


    if not os.path.exists('test_embeddings.npz'):
        mfccs = list()
        for i in tqdm(test_audio_paths):
            mfccs.append(sample_from_mfcc(read_mfcc(i, SAMPLE_RATE), NUM_FRAMES))
        test_mfccs = np.vstack([mfccs])
        predictions = model.m.predict(test_mfccs, batch_size=BATCH_SIZE)
        all_params = OrderedDict([(test_audio_paths[i].split('/')[-1], predictions[i]) for i in range(len(test_audio_paths))])
        np.savez('test_embeddings', **all_params)
    test_predictions = np.load('test_embeddings.npz')


    
    anchor_test=[]
    with tqdm(range(len(test_audio_paths))) as bar:
        for i in bar:
            scores_test=np.zeros(len(speakers))
            test_audio_path=test_audio_paths[i]
            test_predict = test_predictions[test_audio_path.split('/')[-1]]
            # test each utterence's similarity to each speaker in train
            for j in range(len(speakers)):
                for predict in spk_predictions[speakers[j]]:
                    scores_test[j]=max(scores_test[j],batch_cosine_similarity(test_predict, predict))
            # print("anchor speaker: ",np.argmax(scores_test))
            speaker_index=np.argmax(scores_test)+1
            bar.set_description(test_audio_path.split('/')[-1]+'spk:'+str(speaker_index))

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

    anchor_test = inference_model(model=dsm)
    anchor_test = [str(i) for i in anchor_test]  #转str
    anchor_test = '\r'.join(anchor_test)  #逗号分隔
    with open('test_prediction.txt', 'a', encoding='utf8') as f:  #写入
        f.writelines(anchor_test+'\n')


import os
from Constant import WORKING_DIR
if __name__ == '__main__':
    # inference(os.path.join(WORKING_DIR,'train'),os.path.join(CHECKPOINTS_TRIPLET_DIR,'ResCNN_checkpoint_1.h5'))
    inference(WORKING_DIR,checkpoint_file='ResCNN_checkpoint_401.h5')
