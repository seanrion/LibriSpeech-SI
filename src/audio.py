import logging
import os
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from python_speech_features import fbank
from tqdm import tqdm

from Constant import SAMPLE_RATE, NUM_FBANKS,SILENCE_THRESHOLD_FILE
from utils import find_files, ensures_dir, ensure_dir_for_filename

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)

def read_mfcc(input_filename, sample_rate, silence_threshold=None):
    audio = Audio.read(input_filename, sample_rate)
    energy = np.abs(audio)
    if(silence_threshold==None):
        silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
    # TODO: could use trim_silence() here or a better VAD.
    audio_voice_only = audio[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc

def test_silence_threshold(input_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    return silence_threshold

def test_audio_silence_threshold(audio_dir, sample_rate):
    ensures_dir(audio_dir)
    silence_threshold = dict()
    with tqdm(find_files(audio_dir, ext='flac')) as bar:
        for cache_file in bar:
            if 'noise' not in cache_file:
                silence_threshold[Path(cache_file).stem] = test_silence_threshold(cache_file,sample_rate)
    with open(SILENCE_THRESHOLD_FILE,'w') as f:
        f.write(str(silence_threshold))

class Audio:

    def __init__(self, cache_dir: str, audio_dir: str = None, sample_rate: int = SAMPLE_RATE, ext='flac'):
        self.ext = ext
        self.cache_dir = cache_dir
        self.audio_dir = audio_dir
        if not os.path.exists(SILENCE_THRESHOLD_FILE):
            test_audio_silence_threshold(audio_dir, sample_rate)
        with open(SILENCE_THRESHOLD_FILE) as f:
            self.silence_threshold_dict = eval(f.read())
        if audio_dir is not None:
            self.build_cache(os.path.expanduser(audio_dir), sample_rate)
        self.speakers_to_utterances = defaultdict(dict)
        for cache_file in find_files(self.cache_dir, ext='npy'):
            speaker_id, utterance_id = Path(cache_file).stem.split('_')
            self.speakers_to_utterances[speaker_id][utterance_id] = cache_file

    @property
    def speaker_ids(self):
        return sorted(self.speakers_to_utterances)

    @staticmethod
    def trim_silence(audio, threshold):
        """Removes silence at the beginning and end of a sample."""
        energy = librosa.feature.rms(audio)
        frames = np.nonzero(np.array(energy > threshold))
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        audio_trim = audio[0:0]
        left_blank = audio[0:0]
        right_blank = audio[0:0]
        if indices.size:
            audio_trim = audio[indices[0]:indices[-1]]
            left_blank = audio[:indices[0]]  # slice before.
            right_blank = audio[indices[-1]:]  # slice after.
        return audio_trim, left_blank, right_blank

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f'audio_dir: {audio_dir}.')
        logger.info(f'sample_rate: {sample_rate:,} hz.')
        audio_files = find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        with tqdm(audio_files) as bar:
            for audio_filename in bar:
                bar.set_description(audio_filename)
                self.cache_audio_file(audio_filename, sample_rate)

    def cache_audio_file(self, input_filename, sample_rate):
        cache_filename = str(Path(input_filename).with_suffix('.npy'))
        cache_filename = cache_filename.replace(self.audio_dir, self.cache_dir)
        ensure_dir_for_filename(cache_filename)
        key = Path(input_filename).stem.rstrip('noise')
        if key in self.silence_threshold_dict.keys():
            silence_threshold = self.silence_threshold_dict[key]
        else:
            silence_threshold = None
        if not os.path.isfile(cache_filename):
            try:
                mfcc = read_mfcc(input_filename, sample_rate, silence_threshold)
                np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)


def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
