#!/bin/bash

if [[ ! $(pip freeze | grep torch) ]]; then
    echo "Installing PyTorch from source"
    git clone --branch v0.4.1 --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    pip install pyyaml typing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export MACOSX_DEPLOYMENT_TARGET=10.9
    fi
    pip install .
    cd -
else
    echo "PyTorch already installed"
fi
