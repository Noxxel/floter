
import argparse
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
        X = self.sig(X)
        return X

    def forward(self, X):
        X = self.encode(X)
        X = self.decode(X)
        return X

if __name__ == '__main__':

    print("Starting")

    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
    parser.add_argument('--l1size', type=int, default=100)
    parser.add_argument('--l2size', type=int, default=30)
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int, default=2, help='number of threads for the dataloader')
    parser.add_argument('--debug', action='store_true', help='shrinks the dataset')
    parser.add_argument('--log', action='store_true', help='log during epochs')
    parser.add_argument('--fresh', action='store_true', help='force a fresh start without loading states')
    parser.add_argument('--opath', type=str, default="", help='force a fresh start without loading states')

    parser.add_argument('--n_fft', type=int, default=2**11)
    parser.add_argument('--hop_length', type=int, default=367) #--> fps: 60.0817
    parser.add_argument('--n_mels', type=int, default=128)

    opt = parser.parse_args()
    print(opt)

    n_fft = opt.n_fft
    hop_length = opt.hop_length
    n_mels = opt.n_mels

    encode_size = opt.l1size
    middle_size = opt.l2size
    l_rate = opt.lr
    n_epochs = opt.niter
    num_workers = opt.workers
    batch_size = 1
    device = torch.device("cuda:0")
    DEBUG = opt.debug
    LOG = opt.log
    log_intervall = 200
    # ipath = "./mels_set_f8820_h735_b256"
    ipath = "./mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
    statepath = os.path.join(os.path.join("./out", opt.opath), "vae_b{}_{}".format(n_mels, middle_size))
    print('final output-path: {}'.format(statepath))

    # log parameters
    log_file = os.open(os.path.join(statepath, "params.txt"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    ret = os.write(log_file, opt)
    os.close(log_file)

    dset = SoundfileDataset(ipath=ipath, out_type="ae", normalize=True)

    if DEBUG:
        print('warning, debugging turnned on!')
        dset.data = dset.data[:100]

    tset, vset = dset.get_split(sampler=False, split_size=0.2)

    TLoader = DataLoader(tset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    VLoader = DataLoader(vset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    vae = AutoEncoder(n_mels, encode=encode_size, middle=middle_size)
    vae.to(device)
    lossf = nn.MSELoss()
    lossf.to(device)

    os.makedirs(statepath, exist_ok=True)

    starting_epoch = 0
    state = ''
    if not opt.fresh:
        outf_files = os.listdir(statepath)
        states = [of for of in outf_files if "vae_" in of]
        states.sort()
        if len(states) >= 1:
            state = os.path.join(statepath, states[-1])
            if os.path.isfile(state):
                vae.load_state_dict(torch.load(state)['state_dict'])

    optimizer = optim.Adam(vae.parameters(), lr=l_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.5)

    if not opt.fresh and os.path.isfile(state):
        optimizer.load_state_dict(torch.load(state)['optim'])
        print("successfully loaded %s" % (state))
        loaded_epoch = int(states[-1][4:-3])
        starting_epoch = loaded_epoch+1
        print("resuming with epoch {}".format(starting_epoch))

    print("Beginning Training with {} frequency buckets".format(n_mels))
    for epoch in tqdm(range(starting_epoch, n_epochs), desc='Epoch'):

        train_running_loss = 0.0
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
        
        train_acc = np.array(train_acc).mean()
        train_running_loss /= len(TLoader)

        val_running_loss = 0.0
        val_acc = []
        vae.eval()
        for idx, (X, _) in enumerate(tqdm(VLoader, desc="Evaluation")):
            X = X.squeeze().to(device)
            out = vae(X)
            loss = lossf(out, X)
            val_running_loss += loss.detach().item()
            val_acc.append(np.abs((X - out).detach().cpu()).mean())
            if LOG and idx != 0 and idx % log_intervall == 0:
                tqdm.write("Current acc: {}".format(val_acc[-1]))

        val_acc = np.array(val_acc).mean()
        val_running_loss /= len(VLoader)

        scheduler.step(val_running_loss)

        tqdm.write("Epoch: {:d} | Train Loss: {:.3f} | Val Loss: {:.3f} | Train Diff: {:.3f} | Val Diff: {:.3f}".format(epoch, train_running_loss, val_running_loss, train_acc, val_acc))

        if (epoch)%10 == 0:
            state = {'state_dict':vae.state_dict(), 'optim':optimizer.state_dict()}
            filename = "{}/vae_{:03d}.nn".format(statepath, epoch)
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
