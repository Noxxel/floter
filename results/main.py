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
import torchvision.utils as vutils

import dcgan
from dataset import SoundfileDataset
from AE_any import AutoEncoder
from infogan import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--l1size', type=int, default=64)
    parser.add_argument('--l2size', type=int, default=16)
    parser.add_argument('--cpu', action='store_true', help='disables cuda')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu')
    parser.add_argument('--workers', type=int, default=2, help='number of threads for the dataloader')
    parser.add_argument('--debug', action='store_true', help='shrinks the dataset')
    parser.add_argument('--image_size', type=int, default=512)

    parser.add_argument('--dcgan', action='store_true', help='use dcgan')
    parser.add_argument('--info', action='store_true', help='use infogan')

    parser.add_argument('--ae', action='store_true', help='use ae')
    parser.add_argument('--mel', action='store_true', help='use mel')
    parser.add_argument('--conv', action='store_true', help='use conv')

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=11)

    parser.add_argument('--n_fft', type=int, default=2**11)
    parser.add_argument('--hop_length', type=int, default=367) #--> fps: 60.0817
    parser.add_argument('--n_mels', type=int, default=128)

    opt = parser.parse_args()
    print(opt)

    n_fft = opt.n_fft
    hop_length = opt.hop_length
    n_mels = opt.n_mels

    fps = 60.0817
    nz = None

    ngpu = 1
    num_workers = opt.workers
    device = torch.device("cuda:{}".format(opt.gpu) if not opt.cpu else "cpu")
    log_intervall = 200
    ipath = "./song_in"
    opath = "./song_out"
    dcgan_path = ""
    if opt.ae:
        nz = 16
        dcgan_path = os.path.join("./gan/ae", 'nz_{}_ngf_{}_ndf_{}_bs_{}/'.format(nz, opt.ngf, opt.ndf, opt.batch_size))
    elif opt.mel:
        nz = 128
        dcgan_path = os.path.join("./gan/mel", 'nz_{}_ngf_{}_ndf_{}_bs_{}/'.format(nz, opt.ngf, opt.ndf, opt.batch_size))
    ae_path = "./gan/ae/ae_states/vae_b{}_{}".format(n_mels, opt.l2size)
    conv_path = "./gan/conv"

    os.makedirs(ipath, exist_ok=True)
    os.makedirs(opath, exist_ok=True)
    os.makedirs(dcgan_path, exist_ok=True)
    os.makedirs(ae_path, exist_ok=True)
    os.makedirs(conv_path, exist_ok=True)

    input_songs = os.listdir(ipath)
    input_songs = [x for x in input_songs if x.endswith(".mp3")]
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
    vae = None
    if opt.ae:
        vae = AutoEncoder(n_mels, encode=opt.l1size, middle=opt.l2size)
        files = os.listdir(ae_path)
        states = [f for f in files if "vae_" in f]
        states.sort()
        if not len(states) > 0:
            raise Exception("no states for autoencoder provided!")
        state = os.path.join(ae_path, states[-1])
        if os.path.isfile(state):
            vae.load_state_dict(torch.load(state)['state_dict'])
        vae.to(device)
        vae.eval()

    # load pretrained dcgan
    netG = None
    load_state = ""
    if opt.dcgan:
        netG = dcgan.Generator(ngpu, nz=nz, ngf=opt.ngf).to(device)
        state_files = os.listdir(dcgan_path)
        states = [of for of in state_files if 'net_state_epoch_' in of]
        states.sort()
        if len(states) >= 1:
            load_state = os.path.join(dcgan_path, states[-1])
            if os.path.isfile(load_state):
                tmp_load = torch.load(load_state)
                netG.load_state_dict(tmp_load["netG"])
                print("successfully loaded {}".format(load_state))
                del tmp_load
                netG.to(device)
                netG.eval()

    # if opt.debug:
    #     for original_mel, s in zip(mels, input_songs):
    #         original_mel = original_mel.to(device)
    #         decoded_mel = vae(original_mel / (-80)).detach().cpu()
    #         decoded_mel = decoded_mel * (-80)

    #         plt.figure(figsize=(24, 6))
    #         dsp.specshow(original_mel.detach().cpu().numpy().T, y_axis="mel", x_axis="time", sr=sr)
    #         plt.title("original mel spectrogram")
    #         plt.colorbar(format='%+2.0f dB')
    #         plt.tight_layout()
    #         # plt.show()
    #         plt.savefig(os.path.join(opath, "{}_original.png".format(s[:-4])))
    #         plt.clf()
            
    #         plt.figure(figsize=(24, 6))
    #         dsp.specshow(decoded_mel.numpy().T, y_axis="mel", x_axis="time", sr=sr)
    #         plt.title("reconstructed mel spectrogram")
    #         plt.colorbar(format='%+2.0f dB')
    #         plt.tight_layout()
    #         #plt.show()
    #         plt.savefig(os.path.join(opath, "{}_reconstructed.png".format(s[:-4])))
    #         plt.clf()
    # exit()

    # load infogan
    igan = None
    if opt.info:
        igan = Generator(latent_dim=opt.latent_dim, n_classes=opt.n_classes, code_dim=opt.code_dim, img_size=opt.img_size, channels=opt.channels)
        igan.to(device)

    to_pil = tf.ToPILImage()
    for m, s in zip(mels, tqdm(input_songs, desc="generating videos")):
        
        if opt.debug:
            print("ffmpeg -i {} -i {} -codec copy {}".format(os.path.join(opath, s[:-1]+"mp4"), os.path.join(ipath, s), os.path.join(opath, s[:-4]+"_sound.mp4")))
            exit()

        m = (m / (-80)).to(device)
        if opt.ae:
            m = vae.encode(m)
            vae.cpu()
        
        os.makedirs(os.path.join(opath, "tmp/"), exist_ok=True)
        os.system("rm {}".format(os.path.join(opath, "tmp/*")))
        img = None
        for i, m_step in enumerate(tqdm(m, desc="generating images")):
            m_step = m_step.unsqueeze(0).unsqueeze(2).unsqueeze(2)
            if opt.dcgan:
                img = netG(m_step)
            if opt.info:
                img = igan(m_step)
            vutils.save_image(img, os.path.join(opath, 'tmp/{:06d}.png'.format(i)), normalize=True)

        if os.system("ffmpeg -r {} -f image2 -s {}x{} -i tmp/%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(fps, opt.image_size, opt.image_size, os.path.join(opath, s[:-1]+"mp4"))):
            raise Exception("ffmpeg call failed!")
        if os.system("ffmpeg -i {} -i {} -codec copy {}".format(os.path.join(opath, s[:-1]+"4"), os.path.join(ipath, s), os.path.join(opath, s[:-4]+"_sound.mp4"))):
            raise Exception("ffmpeg call failed!")

    print("done")