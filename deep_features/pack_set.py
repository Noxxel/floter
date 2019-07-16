import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm

n_fft = 2**11
hop_length = 367
#hop_length = 2**9
n_mels = 128

#PATH = "./mels_set_f8820_h735_b256"
PATH = "./mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
timesteps = 1800

if not os.path.isdir(PATH):
    raise RuntimeError(f"{PATH} no such directory!")

def main():
    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [os.path.join(PATH, x['path'][:-4] + '.npy') for x in d.values()]
    l = [x for x in l if os.path.isfile(x)]
    del d
    print(len(l))
    full = []
    
    for p in tqdm(l):
        X = np.load(p)
        #print(X.shape)
        #exit()
        X = X.T[:timesteps,:]
        full.append(X)

    full = np.array(full).reshape(-1, 128)
    print(full.shape)
    np.save(os.path.join(PATH, "full_set.npy"), full.astype(np.float32))

main()