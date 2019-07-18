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
    y_cat[range(y.shape[0]), y] = 1.0

    return torch.tensor(y_cat, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 8  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Softmax(dim=1))
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

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
    
    opt = parser.parse_args()
    print(opt)

    if (opt.ae and opt.mel) or (opt.ae and opt.conv) or (opt.mel and opt.conv):
        raise Exception("only specify one of '--ae', '--mel', '--conv'!")

    n_fft = 2**11
    hop_length = 367
    n_mels = 128

    num_workers = opt.workers
    device = torch.device("cuda:{}".format(opt.gpu) if opt.cuda else "cpu")
    device2 = torch.device("cuda:{}".format(0 if opt.gpu == 1 else 1) if opt.cuda else "cpu")
    device2 = device
    ipath = "../deep_features/mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
    opath = "./out"
    statepath = ""

    if opt.ae:
        statepath = "./states/vae_b{}_{}".format(n_mels, opt.l2size)
    elif opt.conv:
        statepath = "./states/conv"

    os.makedirs(ipath, exist_ok=True)
    os.makedirs(opath, exist_ok=True)
    if statepath != "":
        os.makedirs(statepath, exist_ok=True)

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
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Initialize generator and discriminator
    generator = Generator(latent_dim=opt.latent_dim, n_classes=1, code_dim=opt.code_dim, img_size=opt.image_size, channels=opt.channels)
    discriminator = Discriminator()

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
    # else:
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
    static_z = torch.tensor(np.zeros((1 ** 2, opt.latent_dim)), dtype=torch.float32).to(device)
    static_label = to_categorical(np.array([num for _ in range(1) for num in range(1)]), num_columns=1).to(device)
    static_code = torch.tensor(np.zeros((1 ** 2, opt.code_dim)), dtype=torch.float32).to(device)
    c1 = None
    c2 = None
    if opt.ae:
        c1 = vae.encode(Mset[69].to(device2)).detach().to(device).unsqueeze(0)
        c2 = vae.encode(Mset[42].to(device2)).detach().to(device).unsqueeze(0)
    elif opt.mel:
        c1 = Mset[69].to(device).unsqueeze(0)
        c2 = Mset[42].to(device).unsqueeze(0)
    assert len(Iset) == len(Mset)

    def sample_image(n_row, epoch):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        generator.eval()
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32).to(device)
        static_sample = generator(z, static_label, static_code)
        if not os.path.isdir("images"):
            os.makedirs("images", exist_ok=True)
        save_image(static_sample.detach().cpu(), "images/static_{:03d}.png".format(epoch), nrow=n_row, normalize=True)

        sample1 = generator(static_z, static_label, c1)
        sample2 = generator(static_z, static_label, c2)
        save_image(sample1.detach().cpu(), "images/c1_{:03d}.png".format(epoch), nrow=n_row, normalize=True)
        save_image(sample2.detach().cpu(), "images/c2_{:03d}.png".format(epoch), nrow=n_row, normalize=True)
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
            valid = torch.tensor(np.ones((current_b_size, 1)), dtype=torch.float32).to(device)
            fake = torch.tensor(np.zeros((current_b_size, 1)), dtype=torch.float32).to(device)

            # Configure input
            # labels = to_categorical(labels.numpy(), num_columns=1)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.tensor(np.random.normal(0, 1, (current_b_size, opt.latent_dim)), dtype=torch.float32).to(device)
            label_input = to_categorical(np.random.randint(0, 1, current_b_size), num_columns=1).to(device)
            # code_input = torch.tensor(np.random.uniform(-1, 1, (current_b_size, opt.code_dim)), dtype=torch.float32)
            code_input = None
            if opt.mel:
                code_input = mels
            elif opt.ae:
                code_input = vae.encode(mels).detach()
            elif opt.conv:
                raise Exception("missing")

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

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

            # Sample labels
            sampled_labels = np.random.randint(0, 1, current_b_size)

            # Ground truth labels
            gt_labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)

            # Sample noise, labels and code as generator input
            z = torch.tensor(np.random.normal(0, 1, (current_b_size, opt.latent_dim)), dtype=torch.float32).to(device)
            label_input = to_categorical(sampled_labels, num_columns=1).to(device)
            code_input = torch.tensor(np.random.uniform(-1, 1, (current_b_size, opt.code_dim)), dtype=torch.float32).to(device)

            gen_imgs = generator(z, label_input, code_input)
            pred_label, pred_code = discriminator(gen_imgs)[1:]

            pred_label = pred_label.to(device)
            pred_code = pred_code.to(device)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(pred_code, code_input)
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