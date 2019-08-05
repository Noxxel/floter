import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_small import LSTM
from dataset import SoundfileDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from colorhash import ColorHash as cHash

n_fft = 2**11
#hop_length = 2**9
hop_length = 367
n_mels = 128
n_time_steps = 1800
NORMALIZE = True
batch_size = 1
num_workers = 1
n_layers = 2

datapath = "./mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
modelpath = "./lstm_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
modelName = "lstm_99.nn"

device = "cuda"

dset = SoundfileDataset("./all_metadata.p", ipath=datapath, out_type="mel", normalize=NORMALIZE, n_time_steps=n_time_steps)

tset, vset = dset.get_split(sampler=False)

TLoader = DataLoader(tset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

model = LSTM(n_mels, batch_size, num_layers=n_layers)
model.to(device)

stateD = torch.load(modelName, map_location=device)
model.load_state_dict(stateD['state_dict'])
#print(stateD['optim'])
#exit()

model.hidden = model.init_hidden(device)

model.eval()
total_train = np.zeros(dset.n_classes)
correct_train = np.zeros(dset.n_classes+1)
total_val = np.zeros(dset.n_classes)
correct_val = np.zeros(dset.n_classes+1)

running_acc = 0.0
for X, y in tqdm(TLoader, desc="Training"):
    
    X, y = X.to(device), y.to(device)
    out = model(X)
    running_acc += model.get_accuracy(out, y)
    y = int(y.cpu().squeeze().numpy())
    total_train[y] += 1
    pred = int(np.argmax(out.squeeze().detach().cpu()))
    if pred == y:
        correct_train[y] += 1

running_acc = running_acc / len(TLoader)
print("Train Accuracy: {:.2f}".format(running_acc))

running_acc = 0.0
for X, y in tqdm(VLoader, desc="Validation"):
    X, y = X.to(device), y.to(device)
    out = model(X)
    running_acc += model.get_accuracy(out, y)
    y = int(y.cpu().squeeze().numpy())
    total_val[y] += 1
    pred = int(np.argmax(out.squeeze().detach().cpu()))
    if pred == y:
        correct_val[y] += 1

running_acc = running_acc / len(VLoader)
print("Validation Accuracy: {:.2f}".format(running_acc))

genres = list(range(dset.n_classes))
genres = [dset.idx2lbl[x] for x in genres]
genres_acc = list(range(dset.n_classes))
genres_acc = [dset.idx2lbl[x] for x in genres_acc]
genres_acc.append("Total")
colors = [cHash(g).hex for g in genres]
colors_acc = [cHash(g).hex for g in genres_acc]

plt.figure(figsize=(16, 8))
plt.bar(list(range(dset.n_classes)), total_train, tick_label=genres, color=colors)
plt.title("Genre distribution of training set")
plt.tight_layout()
plt.savefig("train_dist.png")
plt.clf()

total = np.sum(correct_train) / np.sum(total_train)
correct_train[:8] = correct_train[:8] / total_train * 100
correct_train[8] = total * 100
plt.figure(figsize=(16, 8))
plt.bar(list(range(dset.n_classes+1)), correct_train, tick_label=genres_acc, color=colors_acc)
plt.title("Accuracy on training set")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("train_acc.png")
plt.clf()

plt.figure(figsize=(16, 8))
plt.bar(list(range(dset.n_classes)), total_val, tick_label=genres, color=colors)
plt.title("Genre distribution of validation set")
plt.tight_layout()
plt.savefig("val_dist.png")
plt.clf()

total = np.sum(correct_val) / np.sum(total_val)
correct_val[:8] = correct_val[:8] / total_val * 100
correct_val[8] = total * 100
plt.figure(figsize=(16, 8))
plt.bar(list(range(dset.n_classes+1)), correct_val, tick_label=genres_acc, color=colors_acc)
plt.title("Accuracy on validation set")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("val_acc.png")
plt.clf()