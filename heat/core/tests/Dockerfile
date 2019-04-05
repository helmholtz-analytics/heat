FROM debian

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 python3-venv libopenmpi-dev openmpi-bin libhdf5-dev libnetcdf-dev

RUN /bin/bash && \
    python3 -m venv ~/.virtualenvs/heat && \
    source ~/.virtualenvs/heat/bin/activate && \
    pip install --upgrade pip && \
    pip install pytest codecov coverage
