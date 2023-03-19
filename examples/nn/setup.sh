#!/usr/bin/env bash

# Load the required modules
ml PyTorch/1.11-CUDA-11.5
ml torchvision/0.12.0-CUDA-11.5


# make a clean virtuel enviroment
python3 -m venv simcrl
# change to the virt-env
source simcrl/bin/activate
# Here you can install other  required packages
pip install requests
pip install thop
pip install tqdm
pip install pytz
pip install python-dateutil
pip install torchinfo



