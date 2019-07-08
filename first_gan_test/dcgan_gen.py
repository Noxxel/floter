import dcgan

import argparse
import torch
import torchvision.utils as vutils


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', required=False, default='weights/netG_epoch_199.pth', help="path to netG (to continue training)")
    parser.add_argument('--netD', required=False, default='weights/netD_epoch_199.pth', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    netG = dcgan.Generator(ngpu).to(device)
    if opt.cuda:
        netG.load_state_dict(torch.load(opt.netG))
    else:
        netG.load_state_dict(torch.load(opt.netG, map_location='cpu'))
    print(netG)

    netD = dcgan.Discriminator(ngpu).to(device)
    if opt.cuda:
        netD.load_state_dict(torch.load(opt.netD))
    else:
        netD.load_state_dict(torch.load(opt.netD, map_location='cpu'))
    print(netD)

    noise = torch.randn(1, nz, 1, 1, device=device)
    fake = netG(noise)

    vutils.save_image(fake.detach(),
            '%s/fake_out.png' % (opt.outf),
            normalize=True)