import torch
import torchvision.transforms as tf
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SoundfileDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

n_fft = 2**10
n_mels = 128
l_rate = 1e-3
n_epochs = 400
num_workers = 2
batch_size = 1
device = "cuda:1"
DEBUG = False
LOG = False
log_intervall = 50
#ipath = "./mels_set_f8820_h735_b256"
ipath = "./mels_set_f{}_b{}".format(n_fft, n_mels)
statepath = "./vae_b{}".format(n_mels)

class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()

        self.input_size = input_size
        self.encode_size = 64
        self.middle = 10

        self.fc1 = nn.Linear(self.input_size, self.encode_size)
        self.fc2 = nn.Linear(self.encode_size, self.middle)
        self.fc3 = nn.Linear(self.middle, self.encode_size)
        self.fc4 = nn.Linear(self.encode_size, self.input_size)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def encode(self, X):
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        return X
    
    def decode(self, X):
        X = self.fc3(X)
        X = self.relu(X)
        X = self.fc4(X)
        X = self.relu(X)
        X = self.sig(X)
        return X

    def forward(self, X):
        X = self.encode(X)
        X = self.decode(X)
        return X

dset = SoundfileDataset(ipath=ipath, out_type="mel", normalize=True)

if DEBUG:
    dset.data = dset.data[:2000]

tset, vset = dset.get_split(sampler=False, split_size=0.2)
TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

vae = AutoEncoder(n_mels)
lossf = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=l_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

val_loss_list, val_acc_list, epoch_list = [], [], []
lossf.to(device)
vae.to(device)
print("Beginning Training with for {} frequency buckets".format(n_mels))
for epoch in tqdm(range(n_epochs), desc='Epoch'):
    train_running_loss = 0.
    train_acc = []
    vae.train()
    for idx, (X, _) in enumerate(tqdm(TLoader, desc="Training")):
        X = X.squeeze().to(device)
        vae.zero_grad()
        out = vae(X)
        loss = lossf(out, X)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.detach().item()
        train_acc.append(np.abs((X - out).detach().cpu()).sum())
        if LOG and idx != 0 and idx % log_intervall == 0:
            tqdm.write("Current loss: {}".format(train_running_loss/idx))
    train_acc = np.array(train_acc) - np.mean(train_acc)
    train_max = np.abs(train_acc).max()
    train_acc = (train_acc / train_max).mean() * 100
    tqdm.write("Epoch: {:d} | Train Loss: {:.2f} | Train Div: {:.2f}".format(epoch, train_running_loss / len(TLoader), train_acc))
    val_running_loss = 0.0
    val_acc = []
    vae.eval()
    for idx, (X, _) in enumerate(tqdm(VLoader, desc="Validation")):
        X = X.squeeze().to(device)
        out = vae(X)
        loss = lossf(out, X)
        val_running_loss += loss.detach().item()
        val_acc.append(np.abs((X - out).detach().cpu()).sum())
    
    val_acc = np.array(val_acc) - np.mean(val_acc)
    val_max = np.abs(val_acc).max()
    val_acc = (val_acc / val_max).mean() * 100
    scheduler.step(val_running_loss/len(VLoader))
    tqdm.write("Epoch: {:d} | Val Loss: {:.2f} | Val Div: {:.2f}".format(epoch, val_running_loss / len(VLoader), val_acc))
    
    epoch_list.append(epoch)
    val_loss_list.append(val_running_loss / len(VLoader))
    val_acc_list.append(val_acc)

    if (epoch+1)%10 == 0:
        state = {'state_dict':vae.state_dict(), 'optim':optimizer.state_dict(), 'epoch_list':epoch_list, 'val_loss':val_loss_list, 'val_acc': val_acc_list}
        filename = "{}/vae_{:02d}.nn".format(statepath, epoch)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        torch.cuda.empty_cache()

# visualization loss
plt.plot(epoch_list, val_loss_list)
plt.ylim(0, np.max(val_loss_list))
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("VAE_b{}: Loss vs # epochs".format(n_mels))
plt.savefig("{}/val_loss.png".format(statepath))
plt.clf()

# visualization accuracy
plt.plot(epoch_list, val_acc_list, color="red")
plt.ylim(0, 100)
plt.xlabel("# of epochs")
plt.ylabel("Divergence")
plt.title("VAE_b{}: Divergence vs # epochs".format(n_mels))
plt.savefig("{}/val_acc.png".format(statepath))
plt.clf()