#!/usr/bin/env bash

# make a clean virtuel enviroment
python3 -m venv simcrl
# change to the virt-env
source simcrl/bin/activate
# Here you can install other  required packages


pip install torch
pip install torchvision
pip install requests
pip install thop
pip install tqdm
pip install pytz
pip install python-dateutil
pip install torchinfo
pip install pandas
pip install Pillow


