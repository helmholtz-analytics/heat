FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y openmpi-bin openmpi-common libopenmpi2 libopenmpi-dev python3-dev python3-venv openssh-client openssh-server
RUN python3 -m venv ~/.virtualenvs/heat && . ~/.virtualenvs/heat/bin/activate && pip install --upgrade pip && pip install codecov coverage pytest

ENV BASH_ENV=/root/.bashenv
RUN echo "cd /heat" >> /root/.bashenv
RUN echo ". ~/.virtualenvs/heat/bin/activate" >> root/.bashenv
RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bash_profile
RUN echo "[[ -s ~/.bashenv ]] && source ~/.bashenv" >> /root/.bashrc
