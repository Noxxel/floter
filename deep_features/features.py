import librosa
import librosa.display as dsp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTM
from dataset import SoundfileDataset
from tqdm import tqdm

filename = "./relish_it.mp3"
duration = 30
n_fft = 2**12        # shortest human-disting. sound (music)
hop_length = 2**10    # => 75% overlap of frames
n_mels = 256
n_epochs = 400
batch_size = 16
l_rate = 1e-4
DEBUG = True

y, sr = librosa.load(filename, mono=True, duration=duration, sr=44100)

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
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) #window of about 10ms 
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
#mel = plot_melspectogram()
#print(mel.shape)
#mel = torch.tensor(mel.T[:1290,:], dtype=torch.float32).unsqueeze(0)
#print(mel.shape)
#exit()

dset = SoundfileDataset("./all_metadata.p", ipath="./mels_set_high", out_type="mel")
if DEBUG:
    dset.data = dset.data[:2000]

tset, vset = dset.get_split(sampler=False)

TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True, drop_last=True)
VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False, drop_last=True)

model = LSTM(n_mels, batch_size, num_layers=2)
#model(mel)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=l_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

val_loss_list, val_accuracy_list, epoch_list = [], [], []
loss_function.to("cuda")
model.to("cuda")

for epoch in tqdm(range(n_epochs), desc='Epoch'):
    train_running_loss, train_acc = 0.0, 0.0
    

    for X, y in tqdm(TLoader, desc="Training"):
        model.hidden = model.init_hidden()
        X, y = X.cuda(), y.cuda()
        model.zero_grad()
        out = model(X)
        del X
        loss = loss_function(out, y)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += model.get_accuracy(out, y)
        del out
        del y

    tqdm.write("Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f" % (epoch, train_running_loss / len(TLoader), train_acc / len(TLoader)))
    val_running_loss, val_acc = 0.0, 0.0
    model.eval()
    for X, y in tqdm(VLoader, desc="Validation"):
        model.hidden = model.init_hidden()
        X, y = X.cuda(), y.cuda()
        out = model(X)
        del X
        val_loss = loss_function(out, y)
        val_running_loss += val_loss.detach().item()
        val_acc += model.get_accuracy(out, y)
        del out
        del y
    scheduler.step(val_loss)
    tqdm.write("Epoch:  %d | Val Loss %.4f  | Val Accuracy: %.2f"
            % (
                epoch,
                val_running_loss / len(VLoader),
                val_acc / len(VLoader),
            )
        )
    model.train()
    epoch_list.append(epoch)
    val_accuracy_list.append(val_acc / len(VLoader))
    val_loss_list.append(val_running_loss / len(VLoader))

    if (epoch+1)%10 == 0:
        state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch_list':epoch_list, 'val_loss':val_loss_list, 'accuracy':val_accuracy_list}
        filename = "./states/lstm_{:02d}.nn".format(epoch)
        torch.save(state, filename)
        del state
        torch.cuda.empty_cache()
