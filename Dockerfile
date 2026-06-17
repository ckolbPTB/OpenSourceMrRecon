ARG BASE_IMAGE=nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04
#ARG BASE_IMAGE=synerbi/sirf:3.10.0-gpu
FROM ${BASE_IMAGE} as base

ENV http_proxy "http://webproxy.berlin.ptb.de:8080"
ENV https_proxy "http://webproxy.berlin.ptb.de:8080"

ARG DEBIAN_FRONTEND=noninteractive
USER root

# install ubuntu dependencies
COPY ubuntu.sh .
RUN bash ubuntu.sh
RUN rm ubuntu.sh

# install anaconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/Miniforge3.sh && \
     /bin/bash ~/Miniforge3.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -n base -c conda-forge conda
RUN conda init

# create conda environment
COPY recon_environment.yml .
RUN conda env create --file recon_environment.yml

# install mrpro
COPY mrpro.sh .
RUN bash mrpro.sh
RUN rm mrpro.sh

# install sigpy
COPY sigpy.sh .
RUN bash sigpy.sh
RUN rm sigpy.sh

# install bart 1.0
COPY bart.sh .
RUN bash bart.sh
RUN rm bart.sh
ENV BART_TOOLBOX_PATH=/bart-1.0.00
ENV PATH=$BART_TOOLBOX_PATH:$PATH
ENV PYTHONPATH=$BART_TOOLBOX_PATH/python:$PYTHONPATH

# install julia and mrireco.jl
COPY mrireco_jl.sh .
RUN bash mrireco_jl.sh
RUN rm mrireco_jl.sh
ENV PATH=/julia-1.10.11/bin:$PATH
COPY mrireco_jl_pkg.jl .
RUN julia mrireco_jl_pkg.jl
RUN rm mrireco_jl_pkg.jl

# reconstruction code
RUN mkdir /compare_recon_packages
COPY recon_scripts/mrpro_paper_julia.jl /compare_recon_packages
COPY recon_scripts/mrpro_paper_reco.py /compare_recon_packages

ENTRYPOINT ["/bin/bash"]







