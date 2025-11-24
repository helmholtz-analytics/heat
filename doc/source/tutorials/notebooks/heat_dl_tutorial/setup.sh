module --force purge
ml Stages/2025
ml GCC
ml OpenMPI
ml CUDA
ml mpi4py
ml PyTorch
ml torchvision
ml h5py
ml heat





python -m venv bench_env
# change to theevirt-env
source bench_env/bin/activate
pip install --upgrade pip
pip install typing_extensions
pip install tqdm
pip install requests
pip install pillow
pip install tqdm
pip install requests
pip install pyarrow
pip install h5py
pip install pillow
pip install nvidia-ml-py3
pip install pandas
