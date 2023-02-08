FROM nvcr.io/nvidia/pytorch:22.11-py3
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update
RUN apt-get install rename
RUN apt-get install -y sox ffmpeg libeigen3-dev

ENV DOCKER_LIB_PATH='/usr/local/lib'

COPY requirements.txt /tmp
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt