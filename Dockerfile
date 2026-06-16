#ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG BASE_IMAGE=synerbi/sirf:3.10.0-gpu
FROM ${BASE_IMAGE} as base

ENV http_proxy "http://webproxy.berlin.ptb.de:8080"
ENV https_proxy "http://webproxy.berlin.ptb.de:8080"

ARG DEBIAN_FRONTEND=noninteractive
USER root
RUN . /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh

# install ubuntu dependencies
COPY ubuntu.sh .
RUN bash ubuntu.sh
RUN rm ubuntu.sh

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# install ismrmrd
#COPY ismrmrd.sh .
#RUN bash ismrmrd.sh
#RUN rm ismrmrd.sh
#ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# install bart 1.0
COPY bart.sh .
RUN bash bart.sh
RUN rm bart.sh
ENV BART_TOOLBOX_PATH=/home/jovyan/bart-1.0.00
ENV PATH=$BART_TOOLBOX_PATH:$PATH
ENV PYTHONPATH=$BART_TOOLBOX_PATH/python:$PYTHONPATH

# install anaconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/Miniforge3.sh && \
#     /bin/bash ~/Miniforge3.sh -b -p /opt/conda
#ENV PATH=$CONDA_DIR/bin:$PATH
#RUN conda update -n base -c conda-forge conda
#RUN conda init

# create conda environment
#COPY recon_environment.yml .
#RUN conda env create --file recon_environment.yml
#ENV PATH=/opt/conda/bin:$PATH

# install gadgetron and sirf
#COPY sirf_gadgetron.sh .
#RUN bash sirf_gadgetron.sh
#RUN rm sirf_gadgetron.sh
#RUN chmod -R go+rwX /opt/SIRF-SuperBuild/INSTALL
#ENV PATH=/opt/SIRF-SuperBuild/INSTALL/bin:$PATH
#ENV LD_LIBRARY_PATH=/opt/SIRF-SuperBuild/INSTALL/lib:$LD_LIBRARY_PATH
#ENV PYTHONPATH=/opt/SIRF-SuperBuild/INSTALL/python:$PYTHONPATH

# install julia and mrireco.jl
COPY mrireco_jl.sh .
RUN bash mrireco_jl.sh
RUN rm mrireco_jl.sh
ENV PATH=/julia-1.8.3/bin:$PATH
COPY mrireco_jl_pkg.jl .
RUN julia mrireco_jl_pkg.jl
RUN rm mrireco_jl_pkg.jl

# install mrpro
COPY mrpro.sh .
RUN bash mrpro.sh
RUN rm mrpro.sh

ENTRYPOINT ["/bin/bash"]







