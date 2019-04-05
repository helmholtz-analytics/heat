FROM fedora

RUN dnf -y update && \
    dnf -y install @development-tools openmpi-devel hdf5-openmpi-devel netcdf-openmpi-devel

RUN python3 -m venv ~/.virtualenvs/heat && \
    source ~/.virtualenvs/heat/bin/activate && \
    pip install --upgrade pip && \
    pip install pytest codecov coverage
