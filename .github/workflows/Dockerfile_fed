FROM fedora:latest

RUN dnf -y update && dnf -y install @development-tools redhat-rpm-config python3-devel openmpi-devel hdf5-openmpi-devel netcdf-openmpi-devel
RUN echo "module load mpi" >> /root/.bashenv
RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bash_profile
RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bashrc

ENV BASH_ENV=/root/.bashenv

RUN python3 -m venv ~/.virtualenvs/heat && \
    . ~/.virtualenvs/heat/bin/activate && \
    pip install --upgrade pip && \
    pip install pytest codecov coverage

#RUN echo "cd /heat" >> root/.bashenv
RUN echo ". ~/.virtualenvs/heat/bin/activate" >> root/.bashenv
