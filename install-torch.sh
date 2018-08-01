#!/bin/bash

#if [[ ! $(pip freeze | grep torch) ]]; then
    pip uninstall torch
    echo "Installing PyTorch from source"
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    pip install pyyaml typing
    pip install .
    cd -
#else
#    echo "PyTorch already installed"
#fi
