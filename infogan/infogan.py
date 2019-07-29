import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage import io, transform

from tqdm import tqdm
from itertools import cycle

from dataset import SoundfileDataset
from AE_any import AutoEncoder

class DatasetCust(Dataset):
    def __init__(self, data_path, transform = None):
        self.data_path = data_path
        file_list = os.listdir(data_path)
        self.data = [f for f in file_list if f.endswith('.jpg')]
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.data[idx])
        image = io.imread(img_name)
        image = image.transpose((2, 0, 1))
        sample = torch.tensor(image, requires_grad=False)

        if self.transform:
            sample = self.transform(sample)
        
        return sample

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 0.9

    return torch.tensor(y_cat, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, latent_dim, code_dim, img_size, channels, ngf=64):
        super(Generator, self).__init__()
        input_dim = latent_dim + code_dim

        self.ngf = ngf
        # we want to start with 1x1 resolution and upscale from there!
        # hardcoding for img_size=512!!
        # doubling the resolution 9 times -> 1x1 -> 512x512
        # self.init_size = img_size // 32  # Initial size before upsampling

        # self.l1 = nn.Sequential(nn.Linear(input_dim, 8*ngf * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=16*ngf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16*ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=16*ngf, out_channels=14*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(14*ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=14*ngf, out_channels=12*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12*ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=12*ngf, out_channels=10*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10*ngf),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(in_channels=10*ngf, out_channels=8*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(in_channels=8*ngf, out_channels=6*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6*ngf),
            nn.ReLU(True),
            # 128 x 128
            nn.ConvTranspose2d(in_channels=6*ngf, out_channels=4*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(True),
            # 256 x 256
            nn.ConvTranspose2d(in_channels=4*ngf, out_channels=2*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(True),
            # 512 x 512
            nn.ConvTranspose2d(in_channels=2*ngf, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, code):
        gen_input = torch.cat((noise, code), -1).unsqueeze(2).unsqueeze(2)
        # out = self.l1(gen_input)
        # out = out.view(out.shape[0], 8*self.ngf, self.init_size, self.init_size)
        img = self.conv_blocks(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, ndf=16, channels=3):
        super(Discriminator, self).__init__()

        # each block halves the image size
        def discriminator_block(in_filters, out_filters):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.BatchNorm2d(out_filters), nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.conv_blocks = nn.Sequential(
            # input is (channels) x 512 x 512
            nn.Conv2d(in_channels=channels, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf) x 256 x 256
            *discriminator_block(ndf, 2*ndf),
            # 128 x 128
            *discriminator_block(2*ndf, 4*ndf),
            # 64 x 64
            *discriminator_block(4*ndf, 6*ndf),
            # 32 x 32
            *discriminator_block(6*ndf, 8*ndf),
            # 16 x 16
            *discriminator_block(8*ndf, 10*ndf),
            # 8 x 8
            *discriminator_block(10*ndf, 12*ndf),
            # 4 x 4
            *discriminator_block(12*ndf, 14*ndf),
            # 2 x 2
            # nn.Conv2d(in_channels=14*ndf, out_channels=16*ndf, kernel_size=2, stride=2, padding=0, bias=False),
            # 1 x 1
        )

        # Output layers
        self.aux_layer = nn.Sequential(nn.Conv2d(in_channels=14*ndf, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False), nn.Sigmoid())

        self.prep_layer = nn.Sequential(nn.Conv2d(in_channels=14*ndf, out_channels=16*ndf, kernel_size=2, stride=2, padding=0, bias=False))
        self.latent_layer = nn.Sequential(nn.Linear(16*ndf, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        # out = out.view(out.shape[0], -1)
        validity = self.aux_layer(out)
        out = self.prep_layer(out)
        # 'reshape' for linear layers
        out = out.view(out.shape[0], -1)
        latent_code = self.latent_layer(out)

        return validity, latent_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False, default="../data/flowers", help='path to dataset')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=84, help="dimensionality of the latent space")
    parser.add_argument("--code_dim", type=int, default=16, help="latent code")
    # parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--image_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument('--fresh', action='store_true', help='perform a fresh start instead of continuing from last checkpoint')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ae', action='store_true', help='train with autoencoder')
    parser.add_argument('--mel', action='store_true', help='train with raw mel spectograms')
    parser.add_argument('--conv', action='store_true', help='train with genre classification conv layer')
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use for the gan")
    parser.add_argument("--workers", type=int, default=2, help="threads for the dataloaders")
    parser.add_argument("--l1size", type=int, default=64, help="layer sizes of ae")
    parser.add_argument("--l2size", type=int, default=16, help="layer sizes of ae")
    parser.add_argument('--debug', action='store_true', help='smaller dataset')

    parser.add_argument('--n_fft', type=int, default=2**11)
    parser.add_argument('--hop_length', type=int, default=367) # --> fps: 60.0817
    parser.add_argument('--n_mels', type=int, default=128)

    parser.add_argument("--ngf", type=int, default=32, help="generator size multiplier")
    parser.add_argument("--ndf", type=int, default=16, help="discriminator size multiplier")
    
    opt = parser.parse_args()
    print(opt)

    assert(opt.image_size == 512)

    if (opt.ae and opt.mel) or (opt.ae and opt.conv) or (opt.mel and opt.conv):
        raise Exception("only specify one of '--ae', '--mel', '--conv'!")

    n_fft = opt.n_fft
    hop_length = opt.hop_length
    n_mels = opt.n_mels

    num_workers = opt.workers
    device = torch.device("cuda:{}".format(opt.gpu) if opt.cuda else "cpu")
    device2 = torch.device("cuda:{}".format(0 if opt.gpu == 1 else 1) if opt.cuda else "cpu")
    device2 = device
    ipath = "../deep_features/mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
    opath = "./out"
    statepath = ""

    if opt.ae:
        statepath = "./states/ae_n{}_b{}_{}".format(opt.n_fft, n_mels, opt.l2size)
    elif opt.conv:
        statepath = "./states/conv"

    os.makedirs(ipath, exist_ok=True)
    os.makedirs(opath, exist_ok=True)
    if statepath != "":
        os.makedirs(statepath, exist_ok=True)

    # log parameters
    log_file = open(os.path.join(opath, "params.txt"), "w")
    log_file.write(str(opt))
    log_file.close()

    # load pretrained autoencoder
    vae = None
    if opt.ae:
        vae = AutoEncoder(n_mels, encode=opt.l1size, middle=opt.l2size)
        files = os.listdir(statepath)
        states = [f for f in files if "vae_" in f]
        states.sort()
        if not len(states) > 0:
            raise Exception("no states for autoencoder provided!")
        state = os.path.join(statepath, states[-1])
        if os.path.isfile(state):
            vae.load_state_dict(torch.load(state)['state_dict'])
        del state
        vae.to(device2)
        vae.eval()

    # load pretrained conv
    # conv = None
    # if opt.conv:
    #     vae = AutoEncoder(n_mels, encode=opt.l1size, middle=opt.l2size)
    #     files = os.listdir(statepath)
    #     states = [f for f in files if "vae_" in f]
    #     states.sort()
    #     if not len(states) > 0:
    #         raise Exception("no states for autoencoder provided!")
    #     state = os.path.join(statepath, states[-1])
    #     if os.path.isfile(state):
    #         vae.load_state_dict(torch.load(state)['state_dict'])
    #     vae.to(device2)
    #     vae.eval()

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Initialize generator and discriminator
    generator = Generator(latent_dim=opt.latent_dim, code_dim=opt.code_dim, img_size=opt.image_size, channels=opt.channels, ngf=opt.ngf)
    discriminator = Discriminator(ndf=opt.ndf)

    # Initialize weights
    lossD = []
    lossG = []
    lossI = []
    load_state = ""
    starting_epoch = 0
    if not opt.fresh:
        outf_files = os.listdir(opath)
        states = [of for of in outf_files if 'infogan_state_epoch_' in of]
        states.sort()
        if len(states) >= 1:
            load_state = os.path.join(opath, states[-1])
            if os.path.isfile(load_state):
                tmp_load = torch.load(load_state)
                discriminator.load_state_dict(tmp_load["idis"])
                generator.load_state_dict(tmp_load["igen"])
                lossD = tmp_load["lossD"]
                lossG = tmp_load["lossG"]
                lossI = tmp_load["lossI"]
                print("successfully loaded {}".format(load_state))
                starting_epoch = int(states[-1][-6:-3])+1
                print("continueing with epoch {}".format(starting_epoch))
                del tmp_load
            else:
                generator.apply(weights_init_normal)
                discriminator.apply(weights_init_normal)
        else:
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Configure data loaders
    Mset = SoundfileDataset(ipath=ipath, out_type="gan")
    assert Mset
    Mloader = torch.utils.data.DataLoader(Mset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))

    Iset = DatasetCust(opt.dataroot,
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((opt.image_size, opt.image_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    assert Iset

    if opt.debug:
        Mset.data = Mset.data[:100]

    assert len(Iset) > len(Mset)
    Iset.data = Iset.data[:len(Mset)]
    assert len(Iset) == len(Mset)
    Iloader = torch.utils.data.DataLoader(Iset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))

    # Optimizers
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    if os.path.isfile(load_state):
        # tmp_dev = device
        tmp_load = torch.load(load_state)
        optimizer_D.load_state_dict(tmp_load["optimD"])
        # for value in optimizer_D.state.values():
        #     for k, v in value.items():
        #         if isinstance(v, torch.Tensor):
        #             value[k] = v.to(tmp_dev)
        optimizer_G.load_state_dict(tmp_load["optimG"])
        # for value in optimizer_G.state.values():
        #     for k, v in value.items():
        #         if isinstance(v, torch.Tensor):
        #             value[k] = v.to(tmp_dev)
        optimizer_info.load_state_dict(tmp_load["optimI"])
        # for value in optimizer_info.state.values():
        #     for k, v in value.items():
        #         if isinstance(v, torch.Tensor):
        #             value[k] = v.to(tmp_dev)
        del tmp_load

    # Static generator inputs for sampling
    static_z = torch.tensor(np.zeros((4, opt.latent_dim)), dtype=torch.float32).to(device)
    static_code = torch.tensor(np.zeros((1, opt.code_dim)), dtype=torch.float32).to(device)
    fixed_noise = torch.randn(4, opt.code_dim, device=device)
    if opt.ae:
        # sample vectors taken from unsmoothened song "Ed Sheeran - Shape of You.mp3"
        fixed_noise[0] = torch.tensor([2.6654, 0.0000, 2.8484, 0.0000, 3.4987, 0.0000, 2.2018, 2.2429, 1.0855,
        2.3932, 2.8212, 0.0000, 1.3639, 2.7173, 2.6672, 1.1499]) #00682.png
        fixed_noise[1] = torch.tensor([2.9989, 0.0000, 2.4614, 0.0000, 2.5902, 0.0000, 2.1589, 2.7042, 0.9324,
        2.2107, 2.3555, 0.0000, 1.7953, 1.8928, 2.8769, 1.2591]) #00691.png
        fixed_noise[2] = torch.tensor([2.8579, 0.0000, 1.8545, 0.0000, 2.5746, 0.0000, 2.2980, 2.6766, 0.9076,
        2.2311, 2.1345, 0.0000, 1.7587, 1.7266, 2.5189, 1.2771]) #00693.png
        fixed_noise[3] = torch.tensor([2.7027, 0.0000, 1.7005, 0.0000, 2.1748, 0.0000, 2.4242, 2.6952, 0.9972,
        2.2989, 1.9881, 0.0000, 1.7763, 1.1806, 2.0095, 1.3649]) #00695.png
    elif opt.mel:
        # sample vectors taken from unsmoothened song "Ed Sheeran - Shape of You.mp3"
        fixed_noise[0] = torch.tensor([0.6755, 0.5194, 0.3433, 0.2868, 0.4381, 0.5859, 0.5946, 0.6565, 0.6675, 0.3035, 0.2805, 0.5485, 0.6015, 0.6540, 0.3872, 0.2686, 0.3610, 0.6525, 0.6497, 0.4101, 0.2440, 0.2997, 0.6601, 0.6909, 0.7617, 0.7218, 0.7331, 0.7593, 0.8599, 0.8910, 0.8966, 0.8561, 0.8174, 0.8566, 0.8911, 0.8416, 0.7651, 0.8500, 0.9715, 0.9539, 0.8429, 0.7134, 0.7963, 0.8829, 0.8653, 0.8572, 0.7956, 0.8870, 0.9926, 0.8561, 0.8309, 0.8466, 0.8681, 0.8724, 0.8333, 0.8739, 0.8027, 0.7903, 0.8753, 0.8907, 0.9170, 0.8266, 0.8394, 0.8960, 0.8924, 0.8485, 0.8562, 0.8983, 0.9250, 0.8805, 0.8856, 0.8860, 0.9521, 0.9083, 0.9702, 0.9389, 0.9334, 0.9181, 0.9244, 0.9538, 0.9891, 0.9633, 0.9488, 0.9890, 0.9938, 0.9441, 0.9718, 0.9558, 0.9295, 1.0000, 1.0000, 1.0000, 1.0000, 0.9949, 0.9905, 1.0000, 0.9815, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9900, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]) #00682.png
        fixed_noise[1] = torch.tensor([0.7254, 0.5733, 0.4330, 0.4146, 0.4894, 0.3721, 0.3743, 0.4410, 0.4814, 0.4368, 0.4127, 0.5789, 0.6392, 0.7095, 0.5159, 0.4009, 0.4961, 0.7021, 0.6857, 0.5552, 0.3862, 0.4389, 0.7320, 0.7428, 0.7117, 0.6857, 0.6978, 0.7800, 0.8513, 0.7929, 0.7289, 0.7291, 0.7612, 0.7914, 0.8291, 0.7983, 0.7546, 0.8084, 0.8248, 0.8017, 0.7737, 0.7588, 0.7447, 0.7476, 0.8231, 0.8109, 0.7361, 0.7491, 0.7993, 0.7718, 0.7631, 0.7359, 0.7264, 0.7496, 0.7334, 0.7214, 0.7624, 0.7939, 0.8128, 0.8063, 0.7827, 0.7298, 0.7351, 0.7675, 0.7325, 0.7583, 0.7313, 0.7146, 0.6910, 0.6757, 0.6760, 0.7229, 0.7451, 0.7912, 0.7656, 0.7325, 0.7645, 0.7648, 0.7573, 0.6793, 0.7053, 0.7540, 0.7586, 0.7540, 0.7460, 0.7263, 0.7138, 0.7529, 0.7244, 0.7682, 0.7658, 0.7624, 0.7890, 0.7330, 0.6849, 0.6863, 0.7214, 0.7468, 0.7638, 0.7727, 0.7513, 0.7902, 0.7796, 0.7532, 0.7392, 0.7607, 0.8114, 0.7715, 0.7744, 0.7396, 0.7343, 0.7529, 0.7683, 0.8538, 0.7960, 0.7808, 0.8094, 0.8056, 0.7723, 0.7867, 0.7881, 0.7288, 0.7189, 0.7614, 0.7431, 0.7297, 0.7649, 0.9765]) #00691.png
        fixed_noise[2] = torch.tensor([0.7263, 0.6260, 0.4180, 0.4006, 0.4410, 0.3791, 0.3780, 0.4238, 0.4312, 0.4419, 0.4329, 0.4891, 0.6229, 0.6359, 0.5481, 0.4297, 0.5155, 0.5728, 0.5434, 0.5311, 0.3972, 0.4446, 0.6223, 0.6332, 0.5685, 0.6028, 0.6388, 0.5967, 0.5739, 0.5437, 0.5681, 0.5469, 0.5934, 0.7968, 0.7157, 0.6081, 0.6216, 0.6105, 0.6264, 0.5807, 0.6083, 0.6965, 0.6809, 0.6578, 0.6476, 0.6665, 0.6049, 0.6121, 0.6882, 0.7398, 0.7338, 0.7293, 0.7377, 0.7773, 0.7290, 0.7299, 0.7152, 0.7178, 0.7405, 0.7061, 0.6793, 0.6818, 0.6375, 0.6557, 0.6934, 0.7314, 0.6945, 0.6582, 0.6966, 0.6833, 0.6874, 0.7419, 0.7097, 0.6971, 0.7163, 0.6622, 0.6934, 0.7161, 0.7146, 0.6806, 0.6938, 0.6567, 0.7278, 0.6824, 0.6761, 0.6298, 0.6745, 0.6705, 0.6427, 0.6790, 0.6787, 0.6782, 0.6998, 0.6567, 0.6182, 0.6084, 0.6801, 0.6581, 0.6587, 0.6740, 0.6625, 0.6841, 0.7022, 0.6977, 0.6877, 0.7184, 0.7466, 0.7234, 0.6927, 0.6821, 0.6886, 0.6985, 0.7326, 0.7565, 0.6882, 0.6971, 0.7170, 0.7164, 0.7031, 0.7178, 0.7129, 0.6584, 0.6482, 0.6740, 0.6609, 0.6633, 0.7094, 0.9176]) #00693.png
        fixed_noise[3] = torch.tensor([0.6553, 0.5815, 0.3396, 0.2950, 0.4415, 0.4918, 0.4662, 0.4194, 0.3829, 0.4063, 0.4311, 0.4665, 0.5471, 0.5856, 0.5265, 0.4607, 0.5399, 0.5500, 0.5217, 0.5348, 0.4382, 0.5073, 0.6203, 0.5995, 0.5323, 0.5586, 0.5394, 0.5353, 0.5188, 0.5023, 0.5080, 0.4855, 0.4946, 0.5044, 0.4950, 0.4852, 0.5021, 0.5010, 0.4975, 0.4741, 0.4684, 0.4809, 0.5169, 0.4963, 0.4653, 0.4362, 0.4158, 0.4332, 0.5003, 0.6285, 0.6862, 0.6887, 0.6160, 0.5630, 0.5221, 0.5290, 0.5031, 0.4748, 0.4708, 0.4832, 0.4740, 0.4532, 0.4886, 0.5066, 0.5224, 0.5929, 0.5714, 0.5768, 0.6439, 0.6377, 0.6205, 0.6595, 0.6358, 0.5491, 0.5085, 0.5115, 0.5411, 0.6030, 0.5865, 0.5980, 0.5939, 0.6260, 0.5000, 0.4780, 0.4762, 0.5009, 0.5617, 0.5990, 0.5382, 0.5370, 0.5713, 0.7067, 0.7343, 0.6448, 0.5795, 0.5688, 0.5778, 0.6117, 0.6002, 0.5899, 0.6203, 0.5154, 0.5081, 0.5671, 0.6601, 0.6426, 0.6238, 0.6256, 0.5679, 0.5428, 0.5564, 0.5721, 0.6646, 0.5920, 0.6201, 0.6714, 0.6578, 0.6369, 0.6521, 0.6950, 0.6700, 0.6750, 0.6435, 0.6143, 0.6335, 0.6204, 0.6852, 0.9004]) #00695.png
    assert len(Iset) == len(Mset)

    def sample_image(n_row, epoch):
        # Static sample
        generator.eval()
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32).to(device)
        static_sample = generator(z, static_code).detach().cpu()
        if not os.path.isdir("images"):
            os.makedirs("images", exist_ok=True)
        save_image(static_sample, "images/randomstatic_{:03d}.png".format(epoch), nrow=n_row, normalize=True)

        sample = generator(static_z, fixed_noise).detach().cpu()
        save_image(sample, "images/fixed_{:03d}.png".format(epoch), nrow=4, normalize=True)
        generator.train()

    # ----------
    #  Training
    # ----------

    generator.cpu()
    discriminator.cpu()
    for epoch in tqdm(range(starting_epoch, opt.n_epochs)):
        torch.cuda.empty_cache()

        generator.to(device)
        discriminator.to(device)

        running_D = 0
        running_G = 0
        running_I = 0
        for i, (real_imgs, mels) in enumerate(tqdm(zip(Iloader, Mloader), total=len(Iloader))):
            assert real_imgs.shape[0] == mels.shape[0]
            current_b_size = real_imgs.shape[0]

            mels = mels.to(device2)
            real_imgs = real_imgs.to(device)

            # Adversarial ground truths
            valid = torch.tensor(np.full((current_b_size, 1), fill_value=0.9), dtype=torch.float32).to(device).unsqueeze(2).unsqueeze(2)
            fake = torch.tensor(np.zeros((current_b_size, 1)), dtype=torch.float32).to(device).unsqueeze(2).unsqueeze(2)

            # Configure input
            # labels = to_categorical(labels.numpy(), num_columns=1)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.tensor(np.random.normal(0, 1, (current_b_size, opt.latent_dim)), dtype=torch.float32).to(device)
            # code_input = torch.tensor(np.random.uniform(-1, 1, (current_b_size, opt.code_dim)), dtype=torch.float32)
            code_input = None
            if opt.mel:
                code_input = mels
            elif opt.ae:
                code_input = vae.encode(mels).detach()
            elif opt.conv:
                raise Exception("missing")

            # Generate a batch of images
            gen_imgs = generator(z, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs)[0].to(device)
            g_loss = adversarial_loss(validity, valid)
            running_G += g_loss.item()

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            real_pred = discriminator(real_imgs)[0].to(device)
            d_real_loss = adversarial_loss(real_pred, valid)


            # Loss for fake images
            fake_pred = discriminator(gen_imgs.detach())[0].to(device)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            running_D += d_loss.item()

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------
            optimizer_info.zero_grad()

            # Sample noise, labels and code as generator input
            z = torch.tensor(np.random.normal(0, 1, (current_b_size, opt.latent_dim)), dtype=torch.float32).to(device)
            code_input = torch.tensor(np.random.uniform(-1, 1, (current_b_size, opt.code_dim)), dtype=torch.float32).to(device)

            gen_imgs = generator(z, code_input)
            pred_code = discriminator(gen_imgs)[1]

            pred_code = pred_code.to(device)

            # info_loss = lambda_con * continuous_loss(pred_code, code_input)
            info_loss = 1 * continuous_loss(pred_code, code_input)
            running_I += info_loss.item()

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress
            # --------------
            tqdm.write("[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [D loss: {:.3f}] [G loss: {:.3f}] [info loss: {:.3f}]".format(epoch, opt.n_epochs, i, len(Iloader), d_loss.item(), g_loss.item(), info_loss.item()))

            del gen_imgs
            del real_imgs

        sample_image(n_row=1, epoch=epoch)
            
        running_D /= len(Iloader)
        running_G /= len(Iloader)
        running_I /= len(Iloader)

        lossD.append(running_D)
        lossG.append(running_G)
        lossI.append(running_I)

        # save state
        discriminator.cpu()
        generator.cpu()

        state = {'idis':discriminator.state_dict(), 'igen':generator.state_dict(), 'optimD':optimizer_D.state_dict(), 'optimG':optimizer_G.state_dict(), 'optimI':optimizer_info.state_dict(), 'lossD':lossD, 'lossG':lossG, 'lossI':lossI}
        filename = os.path.join(opath, "infogan_state_epoch_{:0=3d}.nn".format(epoch))
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        del state