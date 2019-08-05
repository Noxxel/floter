#please excuse the dirty code if you read it before i managed to clean it up

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=16*ngf, kernel_size=4, stride=1, padding=0, bias=False),
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
            nn.ConvTranspose2d(in_channels=2*ngf, out_channels=nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(in_channels=ndf, out_channels=2*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 128 x 128
            nn.Conv2d(in_channels=2*ndf, out_channels=4*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 64 x 64
            nn.Conv2d(in_channels=4*ndf, out_channels=6*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*6) x 32 x 32
            nn.Conv2d(in_channels=6*ndf, out_channels=8*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(in_channels=8*ndf, out_channels=10*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*10) x 8 x 8
            nn.Conv2d(in_channels=10*ndf, out_channels=12*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*12) x 4 x 4
            nn.Conv2d(in_channels=12*ndf, out_channels=14*ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(14*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*14) x 2 x 2
            nn.Conv2d(in_channels=14*ndf, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False),
            # nn.AdaptiveAvgPool2d((1,1)),
            # batchsize x 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
