#!/bin/sh

# 1. If necessary, install openmpi
# Heat can also be installed with pip, but ```openmpi``` has to be available on the system. To install ```openmpi``` on linux/macos:

# Ubuntu
# sudo apt install openmpi-bin libopenmpi-dev

# Arch
# sudo pacman -S openmpi

# MacOS
# brew install openmpi

# 2. Create environment and install dependencies
python -m venv heat-env
source heat-env/bin/activate
pip install heat xarray jupyter scikit-learn ipyparallel

# 3. Setup jupyter kernel
python -m ipykernel install --user --name=heat-env

# 4. Start jupyter
jupyter notebook
