import librosa
import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm

INPATH  = "./fma_small"
OUTPATH = "./mels_set_db"

def save_mel(paths):
    
    inpath, outpath = paths
    
    try:

        song, sr = librosa.load(inpath, mono=True, sr=44100)

        n_fft = 2**11         # shortest human-disting. sound (music)
        hop_length = 2**9    # => 75% overlap of frames
        n_mels = 256

        if len(song) < n_fft:
            return

        X = librosa.feature.melspectrogram(song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        X = librosa.power_to_db(X, ref=np.max)
        if not os.path.isdir(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        
        np.save(outpath, X.astype(np.float32))
        
    except Exception as e:
        
        print(e)
        return
        

    return

def main():
    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [(os.path.join(INPATH, x['path']), os.path.join(OUTPATH, x['path'][:-4] + '.npy')) for x in d.values()]
    l = [x for x in l if os.path.isfile(x[0]) and not os.path.isfile(x[1])]
    del d

    pool = multiprocessing.Pool()
    imap = pool.imap(save_mel, l) 
    l = [x for x in tqdm(imap, total=len(l))]

if __name__ == '__main__':
    main()
