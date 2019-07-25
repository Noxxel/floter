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
        conv = LSTM(n_mels)
        files = os.listdir(statepath)
        states = [f for f in files if "lstm_" in f]
        states.sort()
        if not len(states) > 0:
            raise Exception("no states for crnn provided!")
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

        # sample vectors taken from unsmoothened song "Ed Sheeran - Shape of You.mp3"
        fixed_noise[0] = torch.tensor([[[[0.6755]], [[0.5194]], [[0.3433]], [[0.2868]], [[0.4381]], [[0.5859]], [[0.5946]], [[0.6565]], [[0.6675]], [[0.3035]], [[0.2805]], [[0.5485]], [[0.6015]], [[0.6540]], [[0.3872]], [[0.2686]], [[0.3610]], [[0.6525]], [[0.6497]], [[0.4101]], [[0.2440]], [[0.2997]], [[0.6601]], [[0.6909]], [[0.7617]], [[0.7218]], [[0.7331]], [[0.7593]], [[0.8599]], [[0.8910]], [[0.8966]], [[0.8561]], [[0.8174]], [[0.8566]], [[0.8911]], [[0.8416]], [[0.7651]], [[0.8500]], [[0.9715]], [[0.9539]], [[0.8429]], [[0.7134]], [[0.7963]], [[0.8829]], [[0.8653]], [[0.8572]], [[0.7956]], [[0.8870]], [[0.9926]], [[0.8561]], [[0.8309]], [[0.8466]], [[0.8681]], [[0.8724]], [[0.8333]], [[0.8739]], [[0.8027]], [[0.7903]], [[0.8753]], [[0.8907]], [[0.9170]], [[0.8266]], [[0.8394]], [[0.8960]], [[0.8924]], [[0.8485]], [[0.8562]], [[0.8983]], [[0.9250]], [[0.8805]], [[0.8856]], [[0.8860]], [[0.9521]], [[0.9083]], [[0.9702]], [[0.9389]], [[0.9334]], [[0.9181]], [[0.9244]], [[0.9538]], [[0.9891]], [[0.9633]], [[0.9488]], [[0.9890]], [[0.9938]], [[0.9441]], [[0.9718]], [[0.9558]], [[0.9295]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[0.9949]], [[0.9905]], [[1.0000]], [[0.9815]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[0.9900]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]], [[1.0000]]]]) #00682.png
        fixed_noise[1] = torch.tensor([[[[0.7254]], [[0.5733]], [[0.4330]], [[0.4146]], [[0.4894]], [[0.3721]], [[0.3743]], [[0.4410]], [[0.4814]], [[0.4368]], [[0.4127]], [[0.5789]], [[0.6392]], [[0.7095]], [[0.5159]], [[0.4009]], [[0.4961]], [[0.7021]], [[0.6857]], [[0.5552]], [[0.3862]], [[0.4389]], [[0.7320]], [[0.7428]], [[0.7117]], [[0.6857]], [[0.6978]], [[0.7800]], [[0.8513]], [[0.7929]], [[0.7289]], [[0.7291]], [[0.7612]], [[0.7914]], [[0.8291]], [[0.7983]], [[0.7546]], [[0.8084]], [[0.8248]], [[0.8017]], [[0.7737]], [[0.7588]], [[0.7447]], [[0.7476]], [[0.8231]], [[0.8109]], [[0.7361]], [[0.7491]], [[0.7993]], [[0.7718]], [[0.7631]], [[0.7359]], [[0.7264]], [[0.7496]], [[0.7334]], [[0.7214]], [[0.7624]], [[0.7939]], [[0.8128]], [[0.8063]], [[0.7827]], [[0.7298]], [[0.7351]], [[0.7675]], [[0.7325]], [[0.7583]], [[0.7313]], [[0.7146]], [[0.6910]], [[0.6757]], [[0.6760]], [[0.7229]], [[0.7451]], [[0.7912]], [[0.7656]], [[0.7325]], [[0.7645]], [[0.7648]], [[0.7573]], [[0.6793]], [[0.7053]], [[0.7540]], [[0.7586]], [[0.7540]], [[0.7460]], [[0.7263]], [[0.7138]], [[0.7529]], [[0.7244]], [[0.7682]], [[0.7658]], [[0.7624]], [[0.7890]], [[0.7330]], [[0.6849]], [[0.6863]], [[0.7214]], [[0.7468]], [[0.7638]], [[0.7727]], [[0.7513]], [[0.7902]], [[0.7796]], [[0.7532]], [[0.7392]], [[0.7607]], [[0.8114]], [[0.7715]], [[0.7744]], [[0.7396]], [[0.7343]], [[0.7529]], [[0.7683]], [[0.8538]], [[0.7960]], [[0.7808]], [[0.8094]], [[0.8056]], [[0.7723]], [[0.7867]], [[0.7881]], [[0.7288]], [[0.7189]], [[0.7614]], [[0.7431]], [[0.7297]], [[0.7649]], [[0.9765]]]]) #00691.png
        fixed_noise[2] = torch.tensor([[[[0.7263]], [[0.6260]], [[0.4180]], [[0.4006]], [[0.4410]], [[0.3791]], [[0.3780]], [[0.4238]], [[0.4312]], [[0.4419]], [[0.4329]], [[0.4891]], [[0.6229]], [[0.6359]], [[0.5481]], [[0.4297]], [[0.5155]], [[0.5728]], [[0.5434]], [[0.5311]], [[0.3972]], [[0.4446]], [[0.6223]], [[0.6332]], [[0.5685]], [[0.6028]], [[0.6388]], [[0.5967]], [[0.5739]], [[0.5437]], [[0.5681]], [[0.5469]], [[0.5934]], [[0.7968]], [[0.7157]], [[0.6081]], [[0.6216]], [[0.6105]], [[0.6264]], [[0.5807]], [[0.6083]], [[0.6965]], [[0.6809]], [[0.6578]], [[0.6476]], [[0.6665]], [[0.6049]], [[0.6121]], [[0.6882]], [[0.7398]], [[0.7338]], [[0.7293]], [[0.7377]], [[0.7773]], [[0.7290]], [[0.7299]], [[0.7152]], [[0.7178]], [[0.7405]], [[0.7061]], [[0.6793]], [[0.6818]], [[0.6375]], [[0.6557]], [[0.6934]], [[0.7314]], [[0.6945]], [[0.6582]], [[0.6966]], [[0.6833]], [[0.6874]], [[0.7419]], [[0.7097]], [[0.6971]], [[0.7163]], [[0.6622]], [[0.6934]], [[0.7161]], [[0.7146]], [[0.6806]], [[0.6938]], [[0.6567]], [[0.7278]], [[0.6824]], [[0.6761]], [[0.6298]], [[0.6745]], [[0.6705]], [[0.6427]], [[0.6790]], [[0.6787]], [[0.6782]], [[0.6998]], [[0.6567]], [[0.6182]], [[0.6084]], [[0.6801]], [[0.6581]], [[0.6587]], [[0.6740]], [[0.6625]], [[0.6841]], [[0.7022]], [[0.6977]], [[0.6877]], [[0.7184]], [[0.7466]], [[0.7234]], [[0.6927]], [[0.6821]], [[0.6886]], [[0.6985]], [[0.7326]], [[0.7565]], [[0.6882]], [[0.6971]], [[0.7170]], [[0.7164]], [[0.7031]], [[0.7178]], [[0.7129]], [[0.6584]], [[0.6482]], [[0.6740]], [[0.6609]], [[0.6633]], [[0.7094]], [[0.9176]]]]) #00693.png
        fixed_noise[3] = torch.tensor([[[[0.6553]], [[0.5815]], [[0.3396]], [[0.2950]], [[0.4415]], [[0.4918]], [[0.4662]], [[0.4194]], [[0.3829]], [[0.4063]], [[0.4311]], [[0.4665]], [[0.5471]], [[0.5856]], [[0.5265]], [[0.4607]], [[0.5399]], [[0.5500]], [[0.5217]], [[0.5348]], [[0.4382]], [[0.5073]], [[0.6203]], [[0.5995]], [[0.5323]], [[0.5586]], [[0.5394]], [[0.5353]], [[0.5188]], [[0.5023]], [[0.5080]], [[0.4855]], [[0.4946]], [[0.5044]], [[0.4950]], [[0.4852]], [[0.5021]], [[0.5010]], [[0.4975]], [[0.4741]], [[0.4684]], [[0.4809]], [[0.5169]], [[0.4963]], [[0.4653]], [[0.4362]], [[0.4158]], [[0.4332]], [[0.5003]], [[0.6285]], [[0.6862]], [[0.6887]], [[0.6160]], [[0.5630]], [[0.5221]], [[0.5290]], [[0.5031]], [[0.4748]], [[0.4708]], [[0.4832]], [[0.4740]], [[0.4532]], [[0.4886]], [[0.5066]], [[0.5224]], [[0.5929]], [[0.5714]], [[0.5768]], [[0.6439]], [[0.6377]], [[0.6205]], [[0.6595]], [[0.6358]], [[0.5491]], [[0.5085]], [[0.5115]], [[0.5411]], [[0.6030]], [[0.5865]], [[0.5980]], [[0.5939]], [[0.6260]], [[0.5000]], [[0.4780]], [[0.4762]], [[0.5009]], [[0.5617]], [[0.5990]], [[0.5382]], [[0.5370]], [[0.5713]], [[0.7067]], [[0.7343]], [[0.6448]], [[0.5795]], [[0.5688]], [[0.5778]], [[0.6117]], [[0.6002]], [[0.5899]], [[0.6203]], [[0.5154]], [[0.5081]], [[0.5671]], [[0.6601]], [[0.6426]], [[0.6238]], [[0.6256]], [[0.5679]], [[0.5428]], [[0.5564]], [[0.5721]], [[0.6646]], [[0.5920]], [[0.6201]], [[0.6714]], [[0.6578]], [[0.6369]], [[0.6521]], [[0.6950]], [[0.6700]], [[0.6750]], [[0.6435]], [[0.6143]], [[0.6335]], [[0.6204]], [[0.6852]], [[0.9004]]]]) #00695.png

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
        del optimizerD
        del optimizerG
        del fixed_noise
        del netD
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

            del fake
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