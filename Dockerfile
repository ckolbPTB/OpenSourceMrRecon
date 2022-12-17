ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# install ubuntu dependencies
COPY ubuntu.sh .
RUN bash ubuntu.sh
RUN rm ubuntu.sh

# install ismrmrd
COPY ismrmrd.sh .
RUN bash ismrmrd.sh
RUN rm ismrmrd.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# install bart 0.7
COPY bart.sh .
RUN bash bart.sh
RUN rm bart.sh
ENV PATH=$/bart-0.7.00:$PATH
ENV TOOLBOX_PATH=/bart-0.7.00
ENV PYTHONPATH=${TOOLBOX_PATH}/python:$PYTHONPATH

# install anaconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -n base -c defaults conda

# create conda environment
COPY recon_environment.yml .
RUN conda env create --file recon_environment.yml
ENV PATH=$/opt/conda/bin:$PATH





