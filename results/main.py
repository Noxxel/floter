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
import subprocess

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
    parser.add_argument('--epoch', type=int, default=-1, help='specify state by epoch')

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

    parser.add_argument('--smooth', action='store_true', help='attempt to smoothen the video by slowly moving in the feature-space')
    parser.add_argument('--smooth_count', type=int, default=5, help='the amount of points in the feature space over which the mean is taken to generate images')

    parser.add_argument('--test', action='store_true', help='small sample song') # 068582.mp3
    parser.add_argument('--song', type=str, default="", help='specify a songname to process just one song')

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
    os.makedirs(dcgan_path, exist_ok=True)
    os.makedirs(ae_path, exist_ok=True)
    os.makedirs(conv_path, exist_ok=True)

    # load pretrained autoencoder
    loaded_epoch = 0
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
            if opt.epoch >= 0:
                states = [st for st in states if "epoch_{}".format(opt.epoch) in st]
                load_state = os.path.join(dcgan_path, states[0])
            if os.path.isfile(load_state):
                tmp_load = torch.load(load_state)
                netG.load_state_dict(tmp_load["netG"])
                print("successfully loaded {}".format(load_state))
                del tmp_load
                netG.to(device)
                netG.eval()
                
                loaded_epoch = int(states[-1][-6:-3])

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
    
    if opt.ae or opt.mel:
        opath = os.path.join(opath, "nz{}_epoch{}_ngf{}_ndf{}_bs{}/".format(nz, int(states[-1][-6:-3]), opt.ngf, opt.ndf, opt.batch_size))
    elif opt.conv:
        raise Exception("fix this!")
    os.makedirs(opath, exist_ok=True)
    # log parameters
    log_file = open(os.path.join(opath, "params.txt"), "w")
    log_file.write(str(opt))
    log_file.close()

    input_songs = [x for x in os.listdir(ipath) if x.endswith(".mp3")]
    if opt.test:
        input_songs = [x for x in input_songs if x == "068582.mp3"]
    elif opt.song != "":
        input_songs = [x for x in input_songs if x == opt.song]
    if not len(input_songs) > 0:
        raise Exception("no input song provided!")
    tmp_list = []
    for s in input_songs:
        exists = False
        for of in os.listdir(opath):
            if of == s:
                exists = True
        if not exists:
            tmp_list.append(s)
    if len(tmp_list) < 1:
        exit()
    for s in input_songs:
        if s not in tmp_list:
            print("found output file, skipping: {}".format(s))

    input_songs = tmp_list
    
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

    for m, s in tqdm(zip(mels, input_songs), desc="generating videos", total=len(input_songs)):
        m = (m / (-80)).to(device)
        if opt.ae:
            vae.to(device)
            m = vae.encode(m)
            vae.cpu()
        
        os.makedirs(os.path.join(opath, "tmp/"), exist_ok=True)
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["rm", "-r", os.path.join(opath, "tmp")], stdout=devnull, stderr=devnull)
        os.makedirs(os.path.join(opath, "tmp/"), exist_ok=True)
        # subprocess.run(["rm", os.path.join(opath, "tmp/*")])

        m_step_history = []
        for i, m_step in enumerate(tqdm(m, desc="generating images for {}".format(s))):
            m_step_cur = m_step.unsqueeze(0).unsqueeze(2).unsqueeze(2)

            if opt.smooth:
                m_step_history.append(m_step_cur)
                while len(m_step_history) > opt.smooth_count:
                    del m_step_history[0]
                meaned_step = torch.mean(torch.stack(m_step_history), dim=0)
                m_step_cur = meaned_step
                m_step_cur.to(device)

            if opt.dcgan:
                img = netG(m_step_cur)
            if opt.info:
                img = igan(m_step_cur)
            vutils.save_image(img, os.path.join(opath, 'tmp/{:06d}.png'.format(i)), normalize=True)

        image_param = os.path.join(opath, "tmp/")+"%06d.png"
        video_no_sound = os.path.join(opath, s[:-1]+"4") if not opt.smooth else os.path.join(opath, "smooth{:02d}_".format(opt.smooth_count)+s[:-1]+"4")
        video_sound = os.path.join(opath, s[:-4]+"_sound.mp4") if not opt.smooth else os.path.join(opath, "smooth{:02d}_".format(opt.smooth_count)+s[:-4]+"_sound.mp4")

        command = ["ffmpeg", "-r", str(fps), "-f", "image2", "-s", str(opt.image_size)+"x"+str(opt.image_size), "-i", image_param, "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", video_no_sound]
        tqdm.write("running system command: {}".format(" ".join(command)))
        subprocess.run(command)

        command = ["ffmpeg", "-i", video_no_sound, "-i", os.path.join(ipath, s), "-codec", "copy", video_sound]
        tqdm.write("running system command: {}".format(" ".join(command)))
        subprocess.run(command)

    with open(os.devnull, 'w') as devnull:
        subprocess.run(["rm", "-r", os.path.join(opath, "tmp")], stdout=devnull, stderr=devnull)