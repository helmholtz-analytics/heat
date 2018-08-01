#!/bin/bash

#if [[ ! $(pip freeze | grep torch) ]]; then
pip uninstall -y torch
echo "Installing PyTorch from source"
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
pip install -q pyyaml typing
pip install -q .
cd -
#else
#    echo "PyTorch already installed"
#fi
