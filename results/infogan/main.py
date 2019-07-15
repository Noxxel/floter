import argparse
import torch
import torchvision.transforms as tf
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import librosa.display as dsp
import numpy as np
from torch.utils.data import DataLoader
import librosa
import cv2

from dataset import SoundfileDataset
from AE_any import AutoEncoder
from infogan import Generator

parser = argparse.ArgumentParser()

parser.add_argument('--l1size', type=int, default=64)
parser.add_argument('--l2size', type=int, default=16)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', type=int, default=0, help='which gpu')
parser.add_argument('--workers', type=int, default=2, help='number of threads for the dataloader')
parser.add_argument('--debug', action='store_true', help='shrinks the dataset')
parser.add_argument('--log', action='store_true', help='log during epochs')

opt = parser.parse_args()
print(opt)

n_fft = 2**11
hop_length = 367
n_mels = 128

num_workers = opt.workers
device = torch.device("cuda:{}".format(opt.gpu) if opt.cuda else "cpu")
log_intervall = 200
ipath = "./song_in"
opath = "./song_out"
statepath = "./states/vae_b{}_{}".format(n_mels, opt.l2size)

os.makedirs(ipath, exist_ok=True)
os.makedirs(opath, exist_ok=True)
os.makedirs(statepath, exist_ok=True)

input_songs = os.listdir(ipath)
if not len(input_songs) > 0:
    raise Exception("no input song provided!")

mels = []
for file in tqdm(input_songs, desc="mel spectograms"):
    if not file.endswith(".mp3"):
        continue
    
    song, sr = librosa.load(os.path.join(ipath, file), mono=True, sr=22050)

    if len(song) < n_fft:
        continue

    X = librosa.feature.melspectrogram(song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    X = librosa.power_to_db(X, ref=np.max)

    mels.append(torch.tensor(X.T, dtype=torch.float32))

# load pretrained autoencoder
vae = AutoEncoder(n_mels, encode=opt.l1size, middle=opt.l2size)
files = os.listdir(statepath)
states = [f for f in files if "vae_" in f]
states.sort()
if not len(states) > 0:
    raise Exception("no states for autoencoder provided!")
state = os.path.join(statepath, states[-1])
if os.path.isfile(state):
    vae.load_state_dict(torch.load(state)['state_dict'])
vae.to(device)
vae.eval()

if opt.debug:
    for original_mel, s in zip(mels, input_songs):
        original_mel = original_mel.to(device)
        decoded_mel = vae(original_mel / (-80)).cpu()
        decoded_mel = decoded_mel * (-80)

        plt.figure(figsize=(24, 6))
        import pdb
        pdb.set_trace()
        dsp.specshow(original_mel.cpu().numpy().T, y_axis="mel", x_axis="time", sr=sr)
        plt.title("original mel spectrogram")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(opath, "{}_original.png".format(s[:-4])))
        plt.clf()
        
        plt.figure(figsize=(24, 6))
        dsp.specshow(decoded_mel.numpy().T, y_axis="mel", x_axis="time", sr=sr)
        plt.title("reconstructed mel spectrogram")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(opath, "{}_reconstructed.png".format(s[:-4])))
        plt.clf()
exit()

# load infogan
igan = Generator(latent_dim=opt.latent_dim, n_classes=opt.n_classes, code_dim=opt.code_dim, img_size=opt.img_size, channels=opt.channels)
igan.to(device)

to_pil = tf.ToPILImage()
for m, s in zip(mels, tqdm(input_songs, desc="generating videos")):
    m = (m / (-80)).to(device)
    encoded_mel = vae.encode(m)
    images = []
    for em in tqdm(encoded_mel, desc="generating images"):
        images.append(igan(em).cpu())
    height, width, layers = images[0].squeeze().transpose(1, 2, 0).shape
    vw = cv2.VideoWriter(os.path.join(opath, s[:-1]+"4"), cv2.VideoWriter_fourcc(*'mp4v'), 60.0817, (width,height), True)
    for image in tqdm(images, desc="writing video"):
        vw.write(to_pil(image.squeeze()))
    vw.release()
    os.system("ffmpeg -i {} -i {} -codec copy {}".format(os.path.join(opath, s[:-1]+"4"), os.path.join(ipath, s), os.path.join(opath, s[:-4]+"_sound.mp4")))
