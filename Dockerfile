FROM continuumio/anaconda3:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /bob
COPY . /bob/
RUN apt update
RUN apt install -y build-essential libopencv-dev python3-opencv ffmpeg libsndfile1 libsndfile-dev wget git tmux

RUN conda env create -f bob.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bob", "/bin/bash", "-c"]

RUN conda config --add channels conda-forge
RUN conda install -y -c conda-forge/label/broken bob.learn.em