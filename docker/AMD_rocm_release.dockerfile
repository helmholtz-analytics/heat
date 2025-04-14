################################################################################################
# Development Environment WIP
# Pulls Pytorch image and installs heat + dependencies
################################################################################################

# Build Arguments
ARG DEFAULT=latest
ARG HEAT_VERSION=1.5.x
ARG ROCM_VERSION=6.2
ARG AMDGPU_VERSION=6.2

ARG VERSION_STRING=rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0


# This is the base image for the release image
FROM rocm/pytorch:${DEFAULT} AS base
COPY ./tzdata.seed /tmp/tzdata.seed
RUN debconf-set-selections /tmp/tzdata.seed

# Starts a new stage for the release image
FROM base AS release-install
# RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install mpi4py --no-binary :all:
# RUN echo ${HEAT_VERSION}
# RUN if [[ ${HEAT_VERSION} =~ ^([1-9]\d*|0)(\.(([1-9]\d*)|0)){2}$ ]]; then \
#        pip install heat[hdf5,netcdf]==${HEAT_VERSION}; \
#    else \
#        pip install heat[hdf5,netcdf]; \
#    fi
