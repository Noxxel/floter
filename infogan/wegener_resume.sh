#!/bin/bash
python infogan.py --cuda --mel --batch_size=4 --code_dim=128 --latent_dim=64 --workers=4
