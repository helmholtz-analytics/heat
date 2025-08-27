```shell
#!/bin/sh

## 1. If necessary, install conda: https://www.anaconda.com/docs/getting-started/miniconda/install


## 2. Setup conda environment
conda create --name heat-env python=3.11
conda activate heat-env || exit 1
conda install -c conda-forge heat xarray jupyter scikit-learn ipyparallel

## 3. Setup kernel
python -m ipykernel install --user --name=heat-env

## 3. Start notebook
jupyter notebook
```
