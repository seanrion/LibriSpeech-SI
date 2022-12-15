
import os
from utils import ensures_dir,find_files
from audio import Audio,test_audio_silence_threshold
from Constant import WORKING_DIR,AUDIO_DIR,SAMPLE_RATE
from tqdm import tqdm
from pathlib import Path
def build_audio_cache(working_dir, audio_dir, sample_rate):
    ensures_dir(audio_dir)
    ensures_dir(working_dir)
    Audio(cache_dir=working_dir, audio_dir=audio_dir, sample_rate=sample_rate)






if __name__ == '__main__':
    # test_audio_silence_threshold(AUDIO_DIR, SAMPLE_RATE)
    build_audio_cache(WORKING_DIR,AUDIO_DIR,SAMPLE_RATE)
