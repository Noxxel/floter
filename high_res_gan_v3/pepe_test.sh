#!/bin/bash
python train.py --cuda --workers=4 --dataroot=../data/pepes --image_size=512 --nz=128 --mel --ngf=64 --ndf=128 --batchSize=6 --lrG=0.00005 --lrD=0.00000625 --override_lr
