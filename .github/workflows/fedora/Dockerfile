FROM fedora:latest

RUN dnf -y update && dnf -y install @development-tools redhat-rpm-config python3-devel openmpi-devel hdf5-openmpi-devel netcdf-openmpi-devel

RUN python3 -m venv /root/.virtualenvs/heat && . /root/.virtualenvs/heat/bin/activate && pip install --upgrade pip && pip install codecov coverage pytest

ENV BASH_ENV=/root/.bashenv
RUN echo ". /root/.virtualenvs/heat/bin/activate" >> root/.bashenv
RUN echo "module load mpi" >> root/.bashenv

RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bash_profile
RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bashrc
