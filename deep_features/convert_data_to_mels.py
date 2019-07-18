import argparse
import librosa
import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm


def save_mel(paths):
    
    inpath, outpath = paths
    
    try:

        song, sr = librosa.load(inpath, mono=True, sr=22050)

        if len(song) < n_fft:
            return

        X = librosa.feature.melspectrogram(song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        X = librosa.power_to_db(X, ref=np.max)

        if not os.path.isdir(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
        
        np.save(outpath, X.astype(np.float32))
        
    except Exception as e:
        
        print(e)
        return
        

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_fft', type=int, default=2**11)
    parser.add_argument('--hop_length', type=int, default=367) #--> fps: 60.0817
    parser.add_argument('--n_mels', type=int, default=128)

    opt = parser.parse_args()
    print(opt)

    n_fft = opt.n_fft
    hop_length = opt.hop_length
    n_mels = opt.n_mels

    ipath  = "./fma_small"
    opath = "./mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)

    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [(os.path.join(ipath, x['path']), os.path.join(opath, x['path'][:-4] + '.npy')) for x in d.values()]
    l = [x for x in l if os.path.isfile(x[0]) and not os.path.isfile(x[1])]
    del d

    pool = multiprocessing.Pool()
    imap = pool.imap(save_mel, l) 
    l = [x for x in tqdm(imap, total=len(l))]