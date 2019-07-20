import argparse
import torch
import os
import random
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np

from tqdm import tqdm
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

import dcgan
from dataset import SoundfileDataset
from AE_any import AutoEncoder
from model import LSTM

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False, default="../data/flowers", help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--image_size', type=int, default=512, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=16)
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--loadstate', default='', help="path to state (to continue training)")
    parser.add_argument('--opath', default='./out/', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--fresh', action='store_true', help='perform a fresh start instead of continuing from last checkpoint')
    parser.add_argument('--ae', action='store_true', help='train with autoencoder')
    parser.add_argument('--mel', action='store_true', help='train with raw mel spectograms')
    parser.add_argument('--conv', action='store_true', help='use input generated from an RCNN')
    
    parser.add_argument('--samples', action='store_true', help='just generate a bunch of samples, no training')
    parser.add_argument("--sample_size", type=int, default=16, help="batch size for samples")
    parser.add_argument("--sample_count", type=int, default=20, help="amount of sample images to generate")

    parser.add_argument("--l1size", type=int, default=64, help="layer sizes of ae")
    parser.add_argument("--l2size", type=int, default=16, help="layer sizes of ae or conv")

    parser.add_argument('--n_fft', type=int, default=2**11)
    parser.add_argument('--hop_length', type=int, default=367) # --> fps: 60.0817
    parser.add_argument('--n_mels', type=int, default=128)

    opt = parser.parse_args()
    print(opt)

    n_fft = opt.n_fft
    hop_length = opt.hop_length
    n_mels = opt.n_mels
    
    n_time_steps = 1800

    statepath = ""
    if opt.ae:
        statepath = "./states/vae_b{}_{}".format(n_mels, opt.l2size)
    elif opt.conv:
        statepath = "./states/conv_b{}_{}".format(n_mels, opt.l2size)

    folder_name = 'nz_{}_ngf_{}_ndf_{}_bs_{}/'.format(opt.nz, opt.ngf, opt.ndf, opt.batchSize)
    opath = os.path.join(opt.opath, folder_name)
    ipath = "../deep_features/mels_set_f{}_h{}_b{}".format(n_fft, hop_length, n_mels)
    
    os.makedirs(opath, exist_ok=True)

    # log parameters
    log_file = open(os.path.join(opath, "params.txt"), "w")
    log_file.write(str(opt))
    log_file.close()

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # dataloaders
    Mset = None
    if opt.conv:
        Mset = SoundfileDataset(ipath=ipath, out_type='cgan', n_time_steps=n_time_steps)
    else:
        Mset = SoundfileDataset(ipath=ipath, out_type="gan")
    assert Mset

    dataset = DatasetCust(opt.dataroot,
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((opt.image_size, opt.image_size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3
    assert dataset

    assert len(dataset) > len(Mset)
    dataset.data = dataset.data[:len(Mset)]
    assert len(Mset) == len(dataset)
    
    Mloader = torch.utils.data.DataLoader(Mset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netD = dcgan.Discriminator(ngpu, ndf=ndf).to(device)
    netD.apply(weights_init)

    netG = dcgan.Generator(ngpu, nz=nz, ngf=ngf).to(device)
    netG.apply(weights_init)

    lrD = []
    lrG = []
    lossD = []
    lossG = []
    load_state = ""
    starting_epoch = 0
    if not opt.fresh:
        outf_files = os.listdir(opath)
        states = [of for of in outf_files if 'net_state_epoch_' in of]
        states.sort()
        if len(states) >= 1:
            load_state = os.path.join(opath, states[-1])
            if os.path.isfile(load_state):
                tmp_load = torch.load(load_state)
                netD.load_state_dict(tmp_load["netD"])
                netG.load_state_dict(tmp_load["netG"])
                lrD = tmp_load["lrD"]
                lrG = tmp_load["lrG"]
                lossD = tmp_load["lossD"]
                lossG = tmp_load["lossG"]
                print("successfully loaded {}".format(load_state))
                starting_epoch = int(states[-1][-6:-3]) + 1
                print("continueing with epoch {}".format(starting_epoch))
                del tmp_load

    if opt.loadstate != '':
        tmp_load = torch.load(load_state)
        netD.load_state_dict(torch.load(opt.loadstate))
        netG.load_state_dict(torch.load(opt.loadstate))
        lrD = tmp_load["lrD"]
        lrG = tmp_load["lrG"]
        lossD = tmp_load["lossD"]
        lossG = tmp_load["lossG"]
        print("successfully loaded {}".format(opt.loadstate))
        starting_epoch = int(states[-1][-6:-3]) + 1
        print("continueing with epoch {}".format(starting_epoch))
        del tmp_load
    
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
            state = torch.load(state)
            vae.load_state_dict(state['state_dict'])
        vae.to(device)
        vae.eval()
        del state
    
    #load pretrained LSTM model
    conv = None
    if opt.conv:
        conv = LSTM(n_mels, opt.batchSize)
        files = os.listdir(statepath)
        states = [f for f in files if "lstm_" in f]
        states.sort()
        if not len(states) > 0:
            raise Exception("no states for autoencoder provided!")
        state = os.path.join(statepath, states[-1])
        if os.path.isfile(state):
            state = torch.load(state)
            conv.load_state_dict(state['state_dict'])
        conv.to(device)
        conv.eval()
        del state
    # print(netG)
    # print(netD)

    criterion = nn.BCELoss()

    fixed_noise = None
    if opt.ae:
        fixed_noise = torch.tensor([vae.encode(Mset[i].to(device)).detach().cpu().numpy() for i in range(1337,1337+opt.batchSize)], dtype=torch.float32).unsqueeze(2).unsqueeze(2).to(device)
        print(fixed_noise.shape)
    elif opt.mel:
        fixed_noise = torch.tensor([Mset[i].numpy() for i in range(1337,1337+opt.batchSize)], dtype=torch.float32).unsqueeze(2).unsqueeze(2).to(device)
    elif opt.conv:
        feature_maps = torch.tensor([conv.convolve(Mset[i].to(device).unsqueeze(0)).detach().squeeze().cpu().numpy() for i in range(1337,1337+opt.batchSize)])
        rand_index = np.random.randint(0, feature_maps.shape[1], size=opt.batchSize)
        fixed_noise = (feature_maps[range(opt.batchSize), rand_index, :]).unsqueeze(2).unsqueeze(2).to(device)
        del feature_maps
        del rand_index
    else:
        fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if os.path.isfile(load_state):
        tmp_load = torch.load(load_state)
        optimizerD.load_state_dict(tmp_load["optimD"])
        optimizerG.load_state_dict(tmp_load["optimG"])
        lrD = tmp_load["lrD"]
        lrG = tmp_load["lrG"]
        lossD = tmp_load["lossD"]
        lossG = tmp_load["lossG"]
        del tmp_load
    
    # for pg in optimizerD.param_groups:
    #     pg["lr"] = opt.lrD
    # for pg in optimizerG.param_groups:
    #     pg["lr"] = opt.lr

    # schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, patience=30, factor=0.5)
    # schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, patience=5, factor=0.2)

    if opt.samples:
        netG.eval()
        Mloader = torch.utils.data.DataLoader(Mset, batch_size=opt.sample_size, shuffle=True, num_workers=int(opt.workers))
        for i, mels in enumerate(tqdm(Mloader, total=opt.sample_count)):
            if i >= opt.sample_count:
                break
            batch_size = mels.size(0)

            # train with fake
            noise = None
            if opt.ae:
                noise = vae.encode(mels.to(device)).unsqueeze(2).unsqueeze(2)
            elif opt.mel:
                noise = mels.to(device).unsqueeze(2).unsqueeze(2)
            elif opt.conv:
                noise = conv.convolve(mels.to(device))
                rand_index = np.random.randint(0, noise.shape[1], size=batch_size)
                noise = (noise[range(batch_size), rand_index, :]).unsqueeze(2).unsqueeze(2)
            else:
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            fake = netG(noise)

            vutils.save_image(fake.detach(), os.path.join(opath, 'fake_samples_{:02d}.png'.format(i)), normalize=True)
        exit()


    for epoch in tqdm(range(starting_epoch, opt.niter)):
        torch.cuda.empty_cache()

        netG.to(device)
        netD.to(device)

        running_D = 0
        running_G = 0
        for i, (data, mels) in enumerate(tqdm(zip(dataloader, Mloader), total=len(dataloader))):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = None
            if opt.ae:
                noise = vae.encode(mels.to(device)).unsqueeze(2).unsqueeze(2)
            elif opt.mel:
                noise = mels.to(device).unsqueeze(2).unsqueeze(2)
            elif opt.conv:
                noise = conv.convolve(mels.to(device))
                rand_index = np.random.randint(0, noise.shape[1], size=batch_size)
                noise = (noise[range(batch_size), rand_index, :]).unsqueeze(2).unsqueeze(2)
            else:
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            running_D += errD.cpu().item()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            running_G += errG.cpu().item()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            tqdm.write('[{:d}/{:d}][{:d}/{:d}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f} lrD: {:.2E} lrG: {:.2E}'.format(epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, optimizerD.param_groups[0]["lr"], optimizerG.param_groups[0]["lr"]))

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        os.path.join(opath , 'real_samples.png'),
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        os.path.join(opath , 'fake_samples_epoch_{:03d}.png'.format(epoch)),
                        normalize=True)
            del real_cpu
            del fake
            del noise
        
        running_D /= len(dataloader)
        running_G /= len(dataloader)

        lrD.append(optimizerD.param_groups[0]["lr"])
        lrG.append(optimizerG.param_groups[0]["lr"])
        lossD.append(running_D)
        lossG.append(running_G)
        
        # schedulerD.step(running_D)
        # schedulerG.step(running_G)

        netG.cpu()
        netD.cpu()

        # save state
        state = {'netD':netD.state_dict(), 'netG':netG.state_dict(), 'optimD':optimizerD.state_dict(), 'optimG':optimizerG.state_dict(), 'lrD':lrD, 'lrG':lrG, 'lossD':lossD, 'lossG':lossG}
        filename = os.path.join(opath, "net_state_epoch_{:0=3d}.nn".format(epoch))
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        del state