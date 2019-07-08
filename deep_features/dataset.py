#!/bin/env python3
import torch
import random
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
import pickle
from tqdm import tqdm
import numpy as np
import librosa

class SoundfileDataset(Dataset):

    def __init__(self, path, ipath="./fma_small", hotvec=True, out_type='raw', mel_seg_size=2**11):
        _, ext = os.path.splitext(path)
        if ext == ".p":
            d = pickle.load(open(path, 'rb'))
        else:
            raise RuntimeError(f"{path}: extention '{ext[1:]}' not known")
        
        #np.seterr(all='ignore')
        
        if out_type == 'mel':
            d = {k:v for k,v in d.items() if os.path.isfile(os.path.join(ipath, v['path'][:-3] + "npy"))
                                          and v["track"]["genre_top"] != ""}

        # Generate class-idx-converter
        classes = set()
        for key, val in tqdm(d.items(), desc="build class set"):
            classes.add(val['track']['genre_top'])
        self.idx2lbl = dict(enumerate(classes))
        self.lbl2idx = {v:k for k,v in self.idx2lbl.items()}
        self.n_classes = len(classes)

        # Copy neccecary data into list of named tuples for quick access
        Struct = namedtuple("Data", "id path duration label")
        self.data = []
        for key, val in tqdm(d.items(), desc="build dataset"):
            try:          # |id is actually not needed here
                tmp = Struct(id=key, path=val['path'], duration=int(val["track"]["duration"]),
                             label=self.lbl2idx[ val['track']['genre_top'] ])
                             #labels=[int(x) for x in val['track']['genres_all'][1:-1].split(",")])
            except ValueError as e:
                continue
            
            self.data.append(tmp)

        self.ipath = ipath       # path of image data
        self.hotvec = hotvec     # whether to return labels as one-hot-vec
        self.out_type = out_type # 'raw' or 'mel' or other stuff
        self.mel_seg_size = mel_seg_size

    def calc_entropy(self, song):
        fsize = 1024
        ssize = 512
        
        lenY = song.shape[0]
        lenCut = lenY - (lenY % ssize)
        if(lenY < fsize):
            print("SONG TOO SHORT!!!")
            return np.array([0, 0, 0, 0, 0, 0])

        energy = np.square(song[:lenCut].reshape(-1,ssize))
        energylist = np.concatenate((energy[:-1], energy[1:]), axis=1)

        framelist = energylist.sum(axis=1)
        p = np.nan_to_num(energylist / framelist[:,None]) #whole frame might be 0 causing division by zero
        entropy = -(p * np.nan_to_num(np.log2(p))).sum(axis=1) #same goes for log

        blocksize = []
        blocksize.append(1)

        numbox = []
        numbox.append((np.absolute(song[:-1] - song[1:])).sum() + lenY)
    
        if((lenY % 2) != 0): #double boxsize, half max and min
            uppervalues = (song[:-1].reshape(-1, 2)).max(axis=1)
            lowervalues = (song[:-1].reshape(-1, 2)).min(axis=1)
        else:
            uppervalues = (song.reshape(-1, 2)).max(axis=1)
            lowervalues = (song.reshape(-1, 2)).min(axis=1)

        maxScale = int(np.floor(np.log(lenY) / np.log(2)))
        for scale in range(1, maxScale):
            blocksize.append(blocksize[scale-1]*2)

            numcols = len(uppervalues)
            dummy = (uppervalues - lowervalues).sum() + numcols

            rising = np.less(uppervalues[:-1], lowervalues[1:])
            dummy += ((lowervalues[1:] - uppervalues[:-1]) * rising).sum() #sum where signal is rising

            falling = np.greater(lowervalues[:-1], uppervalues[1:])
            dummy += ((lowervalues[:-1] - uppervalues[1:]) * falling).sum() #sum where signal is falling
            
            numbox.append(dummy/blocksize[scale])

            if((numcols % 2) != 0): #double boxsize, half max and min
                uppervalues = (uppervalues[:-1].reshape(-1, 2)).max(axis=1)
                lowervalues = (lowervalues[:-1].reshape(-1, 2)).min(axis=1)
            else:
                uppervalues = (uppervalues.reshape(-1, 2)).max(axis=1)
                lowervalues = (lowervalues.reshape(-1, 2)).min(axis=1)

        N = np.log(numbox)
        R = np.log(1/np.array(blocksize))
        m = np.linalg.lstsq(R[:,None],N, rcond=None) #slope of curve is fractal dimension

        avg = np.average(entropy)
        std = np.std(entropy)
        mxe = np.max(entropy)
        mne = np.min(entropy)
        entdif = entropy[:-1] - entropy[1:]
        med = max(entdif.min(), entdif.max(), key=abs)
        frd = m[0][0]

        return np.array([avg, std, mxe, mne, med, frd])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.out_type == 'mel':
            return self.get_mel(idx)

        this = self.data[idx]

        try:
            song, sr = librosa.load(os.path.join(self.ipath, this.path))
            
            if self.out_type == 'raw':
                X = song
            elif self.out_type == 'entr':
                X = self.calc_entropy(song)
            else:
                raise ValueError(f"wrong out_type '{self.out_type}'")
        
        except Exception as e:
            print(f"offs:{offs}; dur:{this.duration}; len:{len(song)}; pth:{this.path}")
            raise StopIteration

        del song, sr

        # create hot-vector (if needed)
        if self.hotvec:
            y = torch.zeros(self.n_classes)
            y[this.label] = 1
        else:
            y = this.label

        return torch.as_tensor(X, dtype=torch.float32), y

    def get_mel(self, idx):
        this = self.data[idx]
        X    = np.load(os.path.join(self.ipath, this.path[:-3]) + "npy")
        le   = X.shape[1] - self.mel_seg_size
        if le >= 0:
            offs = np.random.randint(0, le + 1)
            X = X[:,offs:offs+self.mel_seg_size]
        else:
            X = X.pad(((0,0), (0,-le)), 'constant', constant_values=0)
        
        return torch.as_tensor(X, dtype=torch.float32), this.label # no 1hot here
        

    
    def get_split(self, sampler=True):
        validation_split = .2
        shuffle_dataset = True
        random_seed= 4 # chosen by diceroll, 100% random
        # Creating data indices for training and validation splits:
        dataset_size = self.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        if sampler:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            return train_sampler, valid_sampler
        else:
            train_set = Subset(self, train_indices)
            valid_set = Subset(self, val_indices)
            return train_set, valid_set

    def get_indices(self, shuffle=True):
        validation_split = .2
        random_seed= 4 # chosen by diceroll, 100% random  
        
        # Creating data indices for training and validation splits:
        dataset_size = self.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices

    
    
if __name__ == "__main__":

    IPATH = "./fma_small"

    dset = SoundfileDataset("./all_metadata.p", IPATH, hotvec=False, out_type='mel')
    
    print(len(dset))
    X, y = dset[0]
    print(X.shape, X.mean(), y)
    print(dset.idx2lbl[y])
    dset.hotvec = True
    X, y = dset[1]
    print(X.shape, X.mean(), y)
    print(dset.n_classes)

    print("### Benchmarking dataloading speed ###")
    #TODO: compare to training with offline-preprocessed data, to see if preprocessing is bottleneck
    dataloader = DataLoader(dset, num_workers=1, batch_size=1)
    for X, y in tqdm(dataloader):
        tqdm.write(y)
