#!/bin/bash
python train.py --cuda --manualSeed=1337 --batchSize=8 --workers=4 --nz=32 --ngf=64 --ndf=32 --conv --l2size=32 --lr=0.00001

