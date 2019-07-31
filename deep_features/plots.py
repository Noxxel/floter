import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LSTM
from AE_any import AutoEncoder
from tqdm import tqdm
import os

filename = "./relish_it.mp3"
#filename = "./fma_small/000/000002.mp3"
duration = 40

n_fft = 2**11
hop_length = 2**9
#hop_length = 367
n_mels = 128
n_time_steps = 2580
NORMALIZE = True

#datapath = "./mels_set_db"
datapath = "./mels_set_f{}_b{}".format(n_fft, n_mels)
statepath = "./conv_big_b{}_norm".format(n_mels)
#statepath = "conv_small_b128"

device = "cuda"

y, sr = librosa.load(filename, mono=True, duration=duration, sr=22050, offset=220)

print("Sample rate:", sr)
print("Signal:", y.shape)

def plot_signal():
    ticks = []
    for i in range(duration+1):
        ticks.append(sr*i)

    plt.figure(figsize=(24, 6))
    plt.plot(list(range(y.shape[0])), y)
    #plt.xticks(ticks=ticks, labels=list(range(duration+1)))
    plt.axis('off')
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

def plot_encoder():
    stateD = torch.load("ae_490.nn")
    ae = AutoEncoder(n_mels, encode=64, middle=16)
    ae.load_state_dict(stateD['state_dict'])
    ae.eval()
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(24, 6))
    dsp.specshow(mel, y_axis="mel", x_axis="time", sr=sr)
    plt.title("Original mel spectogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig("orig_spec.png")
    plt.clf()
    reconstructed = (ae(torch.tensor(mel.T/(-80), dtype=torch.float32)) * (-80)).detach().numpy().T
    plt.figure(figsize=(24, 6))
    dsp.specshow(reconstructed, y_axis="mel", x_axis="time", sr=sr)
    plt.title("Reconstructed mel spectogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig("recon_spec.png")
    plt.clf()


#plot_signal()

#fft = plot_spectogram()
mel = plot_melspectogram()
#plot_encoder()
exit()
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