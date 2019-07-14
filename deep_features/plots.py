import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LSTM
from tqdm import tqdm
import os

filename = "./relish_it.mp3"
#filename = "./fma_small/000/000002.mp3"
duration = 30

n_fft = 2**11
hop_length = 2**10
#hop_length = 220
n_mels = 256
n_time_steps = 2580
NORMALIZE = True

#datapath = "./mels_set_db"
datapath = "./mels_set_f{}_b{}".format(n_fft, n_mels)
statepath = "./conv_big_b{}_norm".format(n_mels)
#statepath = "conv_small_b128"

device = "cuda"

y, sr = librosa.load(filename, mono=True, duration=duration, sr=22050)

print("Sample rate:", sr)
print("Signal:", y.shape)

def plot_signal():
    ticks = []
    for i in range(duration+1):
        ticks.append(sr*i)

    plt.figure(figsize=(24, 6))
    plt.plot(list(range(y.shape[0])), y)
    plt.xticks(ticks=ticks, labels=list(range(duration+1)))
    #plt.show()
    plt.tight_layout()
    plt.savefig("signal.png")
    plt.clf()

def plot_spectogram():
    fft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    plt.figure(figsize=(24, 6))
    dsp.specshow(librosa.amplitude_to_db(np.abs(fft), ref=np.max), y_axis="log", x_axis="time", sr=sr)
    plt.title("Power spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.show()
    plt.savefig("spectogram.png")
    plt.clf()

    return fft

def plot_melspectogram():
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    plt.figure(figsize=(24, 6))
    dsp.specshow(librosa.power_to_db(mel, ref=np.max), y_axis="mel", x_axis="time", sr=sr)
    plt.title("Mel spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.show()
    plt.savefig("mel_spectogram.png")
    plt.clf()

    return mel

#plot_signal()
#fft = plot_spectogram()
mel = plot_melspectogram()
mel = librosa.power_to_db(mel)
print(mel.shape)

mel = mel.T[:2580,:]
if NORMALIZE:
    mel = mel - mel.mean(axis=0)
    safe_max = np.abs(mel).max(axis=0)
    safe_max[safe_max==0] = 1
    mel = mel / safe_max
print(mel.shape)
#mel = torch.tensor(mel.T[:1290,:], dtype=torch.float32).unsqueeze(0)

model = LSTM(n_mels, 1, num_layers=2)
""" mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
model.hidden = model.init_hidden(device)
model.to(device)
print(mel.shape)
mel = mel.to(device)
model(mel)
exit() """