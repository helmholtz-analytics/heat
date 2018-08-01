#!/bin/bash

if [[ ! $(pip freeze | grep torch) ]]; then
    echo "Installing PyTorch from source"
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule -q update --init
    pip install pyyaml -q
    pip install . -q
    cd -
else
    echo "PyTorch already installed"
fi
