import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTM
from dataset import SoundfileDataset

filename = "/home/flo/Lectures/floter/deep_features/relish_it.mp3"
duration = 30
n_fft = 2**11         # shortest human-disting. sound (music)
hop_length = 2**9    # => 75% overlap of frames
n_mels = 128
n_epochs = 1
batch_size = 16

y, sr = librosa.load(filename, mono=True, duration=duration)

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
    dsp.specshow(librosa.amplitude_to_db(np.abs(fft), ref=np.max), y_axis="log", x_axis="time", sr=sr)
    plt.title("Power spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.show()
    plt.savefig("spectogram.png")
    plt.clf()

    return fft

def plot_melspectogram():
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) #window of about 10ms 
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
print(mel.shape)
dset = SoundfileDataset("./all_metadata.p", out_type="mel")
print("N_Classes:", dset.n_classes)
tset, vset = dset.get_split(sampler=False)
TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True)
VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False)

model = LSTM(n_mels, 128, batch_size)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

val_loss_list, val_accuracy_list, epoch_list = [], [], []

for epoch in range(n_epochs):
    print("Epoch: ", epoch)
    train_running_loss, train_acc = 0.0, 0.0
    model.hidden = model.init_hidden()
    for X, y in TLoader:
        print(X.shape)
        out = model(X)