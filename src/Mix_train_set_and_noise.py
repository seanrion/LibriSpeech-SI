import os
from utils import ensures_dir,find_files
from audio import Audio
from Constant import WORKING_DIR,AUDIO_DIR,SAMPLE_RATE
from pathlib import Path
import numpy as np
from tqdm import tqdm
import soundfile
def produce_train_set_with_noise(audio_dir, sample_rate):
    ensures_dir(audio_dir)
    train_dir = os.path.join(audio_dir,'train')
    noise_dir = os.path.join(audio_dir,'noise')
    ensures_dir(train_dir)
    ensures_dir(noise_dir)
    audio_ext = 'flac'
    noise_ext = 'wav'
    # clean_train_noise_set(train_dir,noise_dir,SAMPLE_RATE,audio_ext)
    produce_train_noise_set(train_dir,noise_dir,SAMPLE_RATE,audio_ext,noise_ext)

def mix_noise_and_audio(input_filename, input_noise_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    noise = Audio.read(input_noise_filename, sample_rate)

    audio_mix = np.zeros(len(audio))

    if(len(audio)<len(noise)):
        time = np.random.choice(range(0, len(noise) - len(audio) + 1))
        noise = noise[time:time+len(audio)]
    for i in range(len(audio)):
        if i < len(noise):
            audio_mix[i] = audio[i] + noise[i]
        else:
            audio_mix[i] = audio[i]
    return audio_mix

def clean_train_noise_set(train_dir, noise_dir, sample_rate, audio_ext):
    audio_files = find_files(train_dir, ext=audio_ext)
    for audio_filename in audio_files:
        if('noise' in audio_filename):
            os.remove(audio_filename)

def produce_train_noise_set(train_dir, noise_dir, sample_rate, audio_ext, noise_ext):
    noise_files = find_files(noise_dir, ext=noise_ext)
    audio_files = find_files(train_dir, ext=audio_ext)
    with tqdm(audio_files) as bar:
        for audio_filename in bar:
            bar.set_description(audio_filename)
            new_stem = Path(audio_filename).stem+'noise'
            noise_filename = np.random.choice(noise_files)
            new_audio = mix_noise_and_audio(audio_filename,noise_filename,sample_rate)
            new_audio_filename = Path(audio_filename).with_stem(new_stem)
            soundfile.write(new_audio_filename,new_audio,sample_rate,format='flac')


if __name__ == '__main__':
    produce_train_set_with_noise(AUDIO_DIR,SAMPLE_RATE)
