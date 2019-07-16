import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

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
        sample = torch.from_numpy(image)

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

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
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
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
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
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

if __name__ == "__main__":
    os.makedirs("images/static/", exist_ok=True)
    os.makedirs("images/varying_c1/", exist_ok=True)
    os.makedirs("images/varying_c2/", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
    parser.add_argument("--code_dim", type=int, default=2, help="latent code")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument('--fresh', action='store_true', help='perform a fresh start instead of continuing from last checkpoint')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ae', action='store_true', help='train with autoencoder')
    parser.add_argument('--mel', action='store_true', help='train with raw mel spectograms')
    parser.add_argument('--conv', action='store_true', help='train with genre classification conv layer')
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use for the gan")
    parser.add_argument("--workers", type=int, default=2, help="threads for the dataloaders")
    
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
    ipath = "../deep_features/mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
    opath = "./out"
    statepath = ""

    if opt.ae:
        statepath = "./states/vae_b{}_{}".format(n_mels, opt.l2size)
    elif opt.conv:
        statepath = "./states/conv"

    os.makedirs(ipath, exist_ok=True)
    os.makedirs(opath, exist_ok=True)
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
    generator = Generator(latent_dim=opt.latent_dim, n_classes=opt.n_classes, code_dim=opt.code_dim, img_size=opt.img_size, channels=opt.channels)
    discriminator = Discriminator()

    if opt.cuda:
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)
        categorical_loss.to(device)
        continuous_loss.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

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
                print("continueing with epoch {}".format(starting_epoch))
                starting_epoch = int(states[-1][-6:-3])

    # Configure data loader
    dataset = DatasetCust(opt.dataroot,
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((opt.img_size, opt.img_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    assert dataset
    Iloader = torch.utils.data.Iloader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))

    # Optimizers
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    
    if os.path.isfile(load_state):
        tmp_load = torch.load(load_state)
        optimizer_D.load_state_dict(tmp_load["optimD"])
        optimizer_G.load_state_dict(tmp_load["optimG"])
        optimizer_info.load_state_dict(tmp_load["optimI"])

    FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

    # Static generator inputs for sampling
    static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
    static_label = to_categorical(np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes)
    static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        static_sample = generator(z, static_label, static_code)
        save_image(static_sample.data, "images/static/{:03d}.png".format(batches_done), nrow=n_row, normalize=True)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
        c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
        sample1 = generator(static_z, static_label, c1)
        sample2 = generator(static_z, static_label, c2)
        save_image(sample1.data, "images/varying_c1/{:03d}.png".format(batches_done), nrow=n_row, normalize=True)
        save_image(sample2.data, "images/varying_c2/{:03d}.png".format(batches_done), nrow=n_row, normalize=True)

    # ----------
    #  Training
    # ----------

    for epoch in tqdm(range(starting_epoch, opt.n_epochs)):
        torch.cuda.empty_cache()
        generator.to(device)
        discriminator.to(device)

        running_D = 0
        running_G = 0
        running_I = 0
        for i, (imgs, mels) in enumerate(zip(cycle(Iloader, Mloader))):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
            running_G += g_loss.item()

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
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
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(pred_code, code_input)
            running_I += info_loss.item()

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress
            # --------------

            tqdm.write("[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [D loss: {:.3f}] [G loss: {:.3f}] [info loss: {:.3f}]".format(epoch, opt.n_epochs, i, len(Iloader), d_loss.item(), g_loss.item(), info_loss.item()))

            batches_done = epoch * len(Iloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
            
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