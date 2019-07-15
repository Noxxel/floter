import dcgan

import argparse
import torch
import os
import random
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from tqdm import tqdm
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


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
    parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=16)
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./out/', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--fresh', action='store_true', help='perform a fresh start instead of continuing from last checkpoint')

    opt = parser.parse_args()
    print(opt)

    folder_name = 'nz_{}_ngf_{}_ndf_{}_bs_{}/'.format(opt.nz, opt.ngf, opt.ndf, opt.batchSize)
    out_path = os.path.join(opt.outf, folder_name)

    try:
        os.makedirs(out_path)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    dataset = DatasetCust(opt.dataroot,
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((opt.imageSize, opt.imageSize)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

    assert dataset
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

    load_state = ""
    starting_epoch = 0
    if not opt.fresh:
        outf_files = os.listdir(out_path)
        states = [of for of in outf_files if 'net_state_epoch_' in of]
        states.sort()
        if len(states) >= 1:
            load_state = os.path.join(out_path, states[-1])
            if os.path.isfile(load_state):
                netD.load_state_dict(torch.load(load_state)["netD"])
                netG.load_state_dict(torch.load(load_state)["netG"])
                print("successfully loaded {}".format(load_state))
                print("continueing with epoch {}".format(starting_epoch))
                starting_epoch = int(states[-1][-6:-3])

    if opt.loadstate != '':
        netD.load_state_dict(torch.load(opt.netD))
        netG.load_state_dict(torch.load(opt.netG))
        print("successfully loaded {}".format(opt.loadstate))
    
    # print(netG)
    # print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if os.path.isfile(load_state):
        optimizerD.load_state_dict(torch.load(load_state)["optimD"])
        optimizerG.load_state_dict(torch.load(load_state)["optimG"])

    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, patience=10, verbose=True, factor=0.25)
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, patience=10, verbose=True, factor=0.25)

    for epoch in tqdm(range(starting_epoch, opt.niter)):
        torch.cuda.empty_cache()
        netG.to(device)
        netD.to(device)

        errD = None
        running_D = 0
        errG = None
        running_G = 0

        for i, data in enumerate(tqdm(dataloader, 0)):
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
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            running_D += errD.item()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            running_G += errG.item()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            tqdm.write('[{:d}/{:d}][{:d}/{:d}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f} lrD: {:.2E} lrG: {:.2E}'.format(epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), optimizerD.param_groups[0]["lr"], optimizerG.param_groups[0]["lr"])

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        os.path.join(out_path , 'real_samples.png'),
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        os.path.join(out_path , 'fake_samples_epoch_{:03d}.png'.format(epoch)),
                        normalize=True)
            del real_cpu
            del fake

        running_D /= len(dataloader)
        running_G /= len(dataloader)
        
        schedulerD.step(errD)
        schedulerG.step(errG)

        netG.to("cpu")
        netD.to("cpu")
        # save state
        state = {'netD':netD.state_dict(), 'netG':netG.state_dict(), 'optimD':optimizerD.state_dict(), 'optimG':optimizerG.state_dict()}
        filename = os.path.join(out_path, "net_state_epoch_{:0=3d}.nn".format(epoch))
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)