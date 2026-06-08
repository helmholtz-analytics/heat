FROM fedora

SHELL ["/bin/bash", "-c"]

RUN dnf -y update && \
    dnf -y install @development-tools redhat-rpm-config python3-devel openmpi-devel hdf5-openmpi-devel netcdf-openmpi-devel

RUN python3 -m venv ~/.virtualenvs/heat && \
    . ~/.virtualenvs/heat/bin/activate && \
    pip install --upgrade pip && \
    pip install pytest codecov coverage

RUN echo "cd /heat && \
    . ~/.virtualenvs/heat/bin/activate && \
    module load mpi" >> /root/.bashrc
