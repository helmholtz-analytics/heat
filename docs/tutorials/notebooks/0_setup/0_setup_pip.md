```shell
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
# shellcheck disable=SC1091
. heat-env/bin/activate || exit 1
pip install heat xarray jupyter scikit-learn ipyparallel

# 3. Setup jupyter kernel
python -m ipykernel install --user --name=heat-env

# 4. Start jupyter
jupyter notebook
```
