from mimetypes import init
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import os
import glob

import matplotlib.pyplot as plt
import requests
from IPython.display import Audio, display

import librosa

import numpy as np

def get_speech_sample(path, resample=None):
    """
    Get waveform of audio sample.

    - path: path of audio sample

    return: waveform
    """
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def play_audio(waveform, sample_rate):
    """
    Play waveform in jupyter notebook.

    - waveform: 1-D waveform
    - sample_rate
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    """
    Plot spectrogram
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    # print(librosa.power_to_db(spec).shape)

    plt.show(block=False)


class AudioProcessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__()
        self.n_mels = n_mels
        # self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms
        self.transform_mfcc = T.MFCC(
                                    sample_rate=sr,
                                    n_mfcc=n_mels,
                                    melkwargs={
                                        "n_fft": n_fft,
                                        "n_mels": n_mels,
                                        "hop_length": self.hop_length,
                                        "mel_scale": "htk",
                                        },
                                    )
        self.transform_mel_spec = T.MelSpectrogram(
                                    sample_rate=sr,
                                    n_fft=n_fft,
                                    win_length=None,
                                    hop_length=self.hop_length,
                                    center=True,
                                    pad_mode="reflect",
                                    power=2.0,
                                    norm="slaney",
                                    onesided=True,
                                    n_mels=n_mels,
                                    mel_scale="htk",
                                )
                                
    def compute_mfccs(self, data):
    
        return self.transform_mfcc(data)

    def mel_spectrogram(self, data):
        
        return self.transform_mel_spec(data)
    

def mfcc_transform(waveform, sample_rate):
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )

    return mfcc_transform(waveform)


def wave_convert_save(wave_path, save_dir, representation):
    """
    Convert wave to spectrogram and save, from .wav to .npy with size 40*101.
    
    - wave_path: path of .wav file
    - save_dir: saving directory
    - representation: "melspec", "mfcc"
    """
    processor = AudioProcessor()
    waveform, sample_rate = get_speech_sample(wave_path)
    in_len = 16000
    # pad to 1 second if not enough wide
    waveform = np.pad(waveform.squeeze(), (0, max(0, in_len - waveform.shape[1])), "constant").reshape(1,-1)
    if representation == 'melspec':
        wave_img = processor.mel_spectrogram(torch.from_numpy(waveform)).numpy()
    else:
        wave_img = processor.compute_mfccs(torch.from_numpy(waveform)).numpy()
    wave_img = librosa.power_to_db(wave_img)
    save_path = os.path.join(save_dir,os.path.basename(wave_path[:-4])+'.npy')
    np.save(save_path, wave_img)


def generate_spec_dataset(
    sound_dir, 
    words_list=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'], 
    representation='melspec'):
    """
    Generate a dataset of spectrograms from waveforms.
    The generated array has shape [1,n_filterbank, 101], default is [1,40,101]

    - sound_dir: '..../speech_commands_v1/audio/'
    - word_list: the words that wanted to be used
    - representation: 'melspec' or 'mfcc'
    """
    img_dir = sound_dir[:-6] + 'imgs/' + representation + '/'
    print("Representation:", representation)

    for file in os.listdir(sound_dir):
        if file in words_list:
            count = 0
            d = os.path.join(sound_dir, file)
            save_dir = os.path.join(img_dir, file)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            wave_paths = glob.glob(d + '/*.wav')
            print("{}: {} files.".format(file, len(wave_paths)))
            for wave_pth in wave_paths:
                wave_convert_save(wave_pth, save_dir, representation)
                count += 1
                if count % 999 == 0:
                    print(count, "wave files converted.")