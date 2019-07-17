print("FUCKTHISFUCKINGSHIT")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTM
from dataset import SoundfileDataset
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

n_fft = 2**11
hop_length = 2**9
n_mels = 128
n_time_steps = 1290
n_layers = 2
NORMALIZE = True

n_epochs = 400
batch_size = 16
l_rate = 1e-3
num_workers = 4

DEBUG = False
LOG = False
log_intervall = 50

#datapath = "./mels_set_db"
datapath = "./mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
statepath = "./lstm_f{}_h{}_b{}_cpu".format(n_fft, hop_length, n_mels)
#statepath = "conv_small_b128"

device = "cuda"

dset = SoundfileDataset("./all_metadata.p", ipath=datapath, out_type="mel", normalize=NORMALIZE, n_time_steps=n_time_steps)
if DEBUG:
    dset.data = dset.data[:2000]

tset, vset = dset.get_split(sampler=False)

TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

model = LSTM(n_mels, batch_size, num_layers=n_layers)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=l_rate)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

stateD = torch.load("lstm_399.nn")
model.load_state_dict(stateD['state_dict'])
val_loss_list, val_accuracy_list, epoch_list = [], [], []
loss_function.to(device)
model.to(device)
model.hidden = model.init_hidden(device)
optimizer.load_state_dict(stateD['optim'])

for epoch in tqdm(range(n_epochs), desc='Epoch'):
    train_running_loss, train_acc = 0.0, 0.0
    model.train()
    for idx, (X, y) in enumerate(tqdm(TLoader, desc="Training")):
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        out = model(X)
        loss = loss_function(out, y)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.detach().item()
        train_acc += model.get_accuracy(out, y)
        if LOG and idx != 0 and idx % log_intervall == 0:
            tqdm.write("Current loss: {}".format(train_running_loss/idx))

    tqdm.write("Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f" % (epoch, train_running_loss / len(TLoader), train_acc / len(TLoader)))
    val_running_loss, val_acc = 0.0, 0.0
    model.eval()
    
    for X, y in tqdm(VLoader, desc="Validation"):
        X, y = X.to(device), y.to(device)
        out = model(X)
        val_loss = loss_function(out, y)
        val_running_loss += val_loss.detach().item()
        val_acc += model.get_accuracy(out, y)

    #scheduler.step(val_loss)
    tqdm.write("Epoch:  %d | Val Loss %.4f  | Val Accuracy: %.2f"
            % (
                epoch,
                val_running_loss / len(VLoader),
                val_acc / len(VLoader),
            )
        )
    
    epoch_list.append(epoch)
    val_accuracy_list.append(val_acc / len(VLoader))
    val_loss_list.append(val_running_loss / len(VLoader))

    if (epoch+1)%10 == 0:
        model.to("cpu")
        state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch_list':epoch_list, 'val_loss':val_loss_list, 'accuracy':val_accuracy_list}
        filename = "{}/lstm_{:02d}.nn".format(statepath, epoch)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        del state
        torch.cuda.empty_cache()
        model.to(device)

# visualization loss
plt.plot(epoch_list, val_loss_list)
plt.ylim(0, np.max(val_loss_list))
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("LSTM: Loss vs # epochs")
plt.savefig("{}/val_loss.png".format(statepath))
plt.clf()

# visualization accuracy
plt.plot(epoch_list, val_accuracy_list, color="red")
plt.ylim(0, 100)
plt.xlabel("# of epochs")
plt.ylabel("Accuracy")
plt.title("LSTM: Accuracy vs # epochs")
plt.savefig("{}/val_acc.png".format(statepath))
plt.clf()
