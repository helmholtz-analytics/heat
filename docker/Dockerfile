ARG PACKAGE_NAME=heat
ARG HEAT_VERSION=1.2.0
ARG HEAT_BRANCH=main
ARG INSTALL_TYPE=release

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel AS base
RUN apt update && apt install -y gnupg2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && apt update && apt install -y openmpi-bin libopenmpi-dev build-essential openssh-client python3-dev git && apt clean && rm -rf /var/lib/apt/lists/*

FROM base AS source-install
ARG HEAT_BRANCH
RUN git clone -b ${HEAT_BRANCH} https://github.com/helmholtz-analytics/heat.git ; cd heat; pip install mpi4py --no-binary :all: ; pip install .[hdf5,netcdf]; pip cache purge ; cd ..; rm -rf heat

FROM base AS release-install
ARG PACKAGE_NAME
ARG HEAT_VERSION
RUN pip install mpi4py --no-binary :all: ; if [ "x${HEAT_VERSION}" = "x" ]; then pip install ${PACKAGE_NAME}[hdf5,netcdf]; else pip install ${PACKAGE_NAME}[hdf5,netcdf]==${HEAT_VERSION}; fi ; pip cache purge

FROM ${INSTALL_TYPE}-install AS final
