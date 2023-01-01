# Constants.
import os
SAMPLE_RATE = 16000  # not higher than that otherwise we may have errors when computing the fbanks.

# Train/Test sets share the same speakers. They contain different utterances.
# 0.8 means 20% of the utterances of each speaker will be held out and placed in the test set.
TRAIN_TEST_RATIO = 0.8

CHECKPOINTS_SOFTMAX_DIR = 'checkpoints-softmax'

CHECKPOINTS_TRIPLET_DIR = 'checkpoints-triplets'

BATCH_SIZE = 32 * 3  # have to be a multiple of 3.

# Input to the model will be a 4D image: (batch_size, num_frames, num_fbanks, 3)
# Where the 3 channels are: FBANK, DIFF(FBANK), DIFF(DIFF(FBANK)).
NUM_FRAMES = 160  # 1 second ~ 100 frames with default params winlen=0.025,winstep=0.01
NUM_FBANKS = 64


WORKING_DIR = os.path.join(os.getcwd(),'cache_dir')
AUDIO_DIR = os.path.join(os.getcwd(),'audio_dir')
SILENCE_THRESHOLD_FILE = 'silence_threshold.txt'

KERAS_DIR = os.path.join(WORKING_DIR,'keras-inputs')
COUNTS_PER_SPEAKER = '600,100'

