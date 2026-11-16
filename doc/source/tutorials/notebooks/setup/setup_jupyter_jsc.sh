#!/bin/bash

source /p/project1/training2655/kernel/activate.sh

ipcontroller  &
srun -n 4 -c 12 --overlap --ntasks-per-node 4 ipengine start
