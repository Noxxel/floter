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

print("Starting")
n_fft = 2**11
n_mels = 256
encode_size = 200
middle_size = 75
l_rate = 1e-3
n_epochs = 400
num_workers = 3
batch_size = 1
device = "cuda:1"
DEBUG = True
LOG = False
log_intervall = 50
ipath = "./mels_set_f8820_h735_b256"
#ipath = "./mels_set_f{}_b{}".format(n_fft, n_mels)
statepath = "./vae_b{}_{}".format(n_mels, middle_size)

class AutoEncoder(nn.Module):
    def __init__(self, input_size, encode=100, middle=25):
        super(AutoEncoder, self).__init__()

        self.input_size = input_size
        self.encode_size = encode
        self.middle = middle

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
    dset.data = dset.data[:1000]

tset = dset.get_train(sampler=False)
TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

vae = AutoEncoder(n_mels, encode=encode_size, middle=middle_size)
lossf = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=l_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

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
        train_acc.append(np.abs((X - out).detach().cpu()).mean())
        if LOG and idx != 0 and idx % log_intervall == 0:
            tqdm.write("Current acc: {}".format(train_acc[-1]))
        import pdb
        pdb.set_trace()
    train_acc = np.array(train_acc)# - np.mean(train_acc)
    train_acc = train_acc.mean()
    # train_max = np.max(train_acc)
    # train_acc = (train_acc / train_max).mean()
    tqdm.write("Epoch: {:d} | Train Loss: {:.2f} | Train Div: {:.2f}".format(epoch, train_running_loss / len(TLoader), train_acc))
    
    epoch_list.append(epoch)

    if (epoch+1)%10 == 0:
        state = {'state_dict':vae.state_dict(), 'optim':optimizer.state_dict(), 'epoch_list':epoch_list, 'val_loss':val_loss_list, 'val_acc': val_acc_list}
        filename = "{}/vae_{:02d}.nn".format(statepath, epoch)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        torch.cuda.empty_cache()

# # visualization loss
# plt.plot(epoch_list, val_loss_list)
# plt.ylim(0, np.max(val_loss_list))
# plt.xlabel("# of epochs")
# plt.ylabel("Loss")
# plt.title("VAE_b{}: Loss vs # epochs".format(n_mels))
# plt.savefig("{}/val_loss.png".format(statepath))
# plt.clf()

# # visualization accuracy
# plt.plot(epoch_list, val_acc_list, color="red")
# plt.ylim(0, 100)
# plt.xlabel("# of epochs")
# plt.ylabel("Divergence")
# plt.title("VAE_b{}: Divergence vs # epochs".format(n_mels))
# plt.savefig("{}/val_acc.png".format(statepath))
# plt.clf()
