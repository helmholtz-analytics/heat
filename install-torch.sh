#!/bin/sh

if [[ $(pip freeze | grep torch) ]]; then
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule -q update --init
    pip install pyyaml -q
    travis_wait 32 pip install . -q
    cd -
fi
