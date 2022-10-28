ARG PACKAGE_NAME=heat
ARG HEAT_VERSION=1.2.0
ARG HEAT_BRANCH=main
ARG INSTALL_TYPE=release

FROM nvcr.io/nvidia/pytorch:22.05-py3 AS base
COPY ./tzdata.seed /tmp/tzdata.seed
RUN debconf-set-selections /tmp/tzdata.seed
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y build-essential openssh-client python3-dev git && apt clean && rm -rf /var/lib/apt/lists/*

FROM base AS source-install
ARG HEAT_BRANCH
RUN git clone -b ${HEAT_BRANCH} https://github.com/helmholtz-analytics/heat.git ; cd heat; pip install mpi4py --no-binary :all: ; pip install .[hdf5,netcdf]; pip cache purge ; cd ..; rm -rf heat

FROM base AS release-install
ARG PACKAGE_NAME
ARG HEAT_VERSION
RUN pip install mpi4py --no-binary :all: ; if [ "x${HEAT_VERSION}" = "x" ]; then pip install ${PACKAGE_NAME}[hdf5,netcdf]; else pip install ${PACKAGE_NAME}[hdf5,netcdf]==${HEAT_VERSION}; fi ; pip cache purge ; true

FROM ${INSTALL_TYPE}-install AS final
