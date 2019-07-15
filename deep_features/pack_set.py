import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm

#PATH = "./mels_set_f8820_h735_b256"
PATH = "./mels_set_f1024_b128"
timesteps = 1798

def main():
    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [os.path.join(PATH, x['path'][:-4] + '.npy') for x in d.values()]
    l = [x for x in l if os.path.isfile(x)]
    del d
    print(len(l))
    full = []
    
    for path in tqdm(l):
        X = np.load(path)
        print(X.shape)
        exit()
        X = X.T[:timesteps,:]
        full.append(X)
    full = np.array(full)
    np.save(os.path.join(PATH, "full_set.npy"), full.astype(np.float32))

if __name__ == '__main__':
    main()