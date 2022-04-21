from mimetypes import init
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import os

import matplotlib.pyplot as plt
import requests
from IPython.display import Audio, display

import librosa

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