import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import namedtuple
from torch.utils.data import Subset
import pickle
from tqdm import tqdm
import numpy as np
import librosa


class SoundfileDataset(Dataset):

    def __init__(self, path="./all_metadata.p", ipath="./mels_set", hotvec=False, out_type='raw', n_time_steps=None, normalize=True):
        _, ext = os.path.splitext(path)
        if ext == ".p":
            d = pickle.load(open(path, 'rb'))
        else:
            raise RuntimeError(f"{path}: extention '{ext[1:]}' not known")
        
        if not os.path.isdir(ipath):
            raise RuntimeError(f"{ipath} no such directory!")
        
        #np.seterr(all='ignore')
        
        if out_type == 'mel' or out_type == 'ae' or out_type == 'gan':
            d = {k:v for k,v in d.items() if os.path.isfile(os.path.join(ipath, v['path'][:-3] + "npy")) and v["track"]["genre_top"] != ""}
        
        # Generate class-idx-converter
        classes = set()
        for key, val in tqdm(d.items(), desc="build class set"):
            classes.add(val['track']['genre_top'])
        classes = sorted(classes)
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

        self.ipath = ipath
        self.hotvec = hotvec     # whether to return labels as one-hot-vec
        self.out_type = out_type
        self.n_time_steps = n_time_steps
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        this = self.data[idx]

        if self.out_type == 'mel' or self.out_type == 'ae' or self.out_type == 'gan' or self.out_type == 'cgan':
            X = np.load(os.path.join(self.ipath, this.path[:-3]) + "npy")
            X = X.T
            if self.out_type == 'cgan':
                X = X[:self.n_time_steps,:]
                if self.normalize:
                    X = X / (-80)
                return torch.as_tensor(X, dtype=torch.float32)
            if self.out_type == 'gan':
                randIndex = np.random.randint(0, X.shape[0])
                X = X[randIndex, :]
                if self.normalize:
                    X = X / (-80)
                return torch.as_tensor(X, dtype=torch.float32)
            else:
                if self.n_time_steps is not None:
                    X = X[:self.n_time_steps,:]
                #normalize data
                if self.normalize and self.out_type == 'mel':
                    X = (X / -80) * 2 - 1 #librosa.power_to_db scales from -80 to 0
                elif self.normalize and self.out_type == 'ae':
                    X = (X / -80) # atuo encoder produces vectors in range 0 to 1
        else:
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
    
    def get_split(self, sampler=True, split_size=0.3):
        validation_split = split_size
        shuffle_dataset = True
        random_seed = 42 # chosen by diceroll, 100% random
        # Creating data indices for training and validation splits:
        dataset_size = self.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:

        train_set = Subset(self, train_indices)
        valid_set = Subset(self, val_indices)
        return train_set, valid_set

    def get_train(self, sampler=True):
        shuffle_dataset = True
        random_seed= 4 # chosen by diceroll, 100% random
        dataset_size = self.__len__()
        indices = list(range(dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        # Creating PT data samplers and loaders:

        train_set = Subset(self, indices)
        return train_set

    def get_indices(self, shuffle=True):
        validation_split = .3
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

    dset = SoundfileDataset(ipath="./mels_set_f2048_h367_b128",out_type='gan', n_time_steps=1800)

    print("### Benchmarking dataloading speed ###")
    dataloader = DataLoader(dset, num_workers=1, batch_size=1)
    minLen = 10000000
    sizes = set()
    for i, [X, y] in enumerate(tqdm(dataloader)):
        sizes.add(X.shape[0])
        if X.shape[0] < minLen:
            minLen = X.shape[0]
    
    print(sizes)
    print(minLen)
