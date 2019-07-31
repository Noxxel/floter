#!/bin/bash
python infogan.py --cuda --ae --batch_size=8 --latent_dim=84 --code_dim=16 --workers=4 --ngf=64 --ndf=32 --n_fft=4096
