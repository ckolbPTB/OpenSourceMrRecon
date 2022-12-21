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

# install bart 0.8
COPY bart.sh .
RUN bash bart.sh
RUN rm bart.sh
ENV TOOLBOX_PATH=/bart-0.8.00
ENV PATH=$TOOLBOX_PATH:$PATH
ENV PYTHONPATH=$TOOLBOX_PATH/python:$PYTHONPATH

# install anaconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -n base -c defaults conda

# create conda environment
COPY recon_environment.yml .
RUN conda env create --file recon_environment.yml
ENV PATH=/opt/conda/bin:$PATH

# install gadgetron and sirf
COPY sirf_gadgetron.sh .
RUN bash sirf_gadgetron.sh
RUN rm sirf_gadgetron.sh
RUN chmod -R go+rwX /opt/SIRF-SuperBuild/INSTALL
ENV PATH=/opt/SIRF-SuperBuild/INSTALL/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/SIRF-SuperBuild/INSTALL/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/SIRF-SuperBuild/INSTALL/python:$PYTHONPATH

# install julia and mrireco.jl
COPY mrireco_jl.sh .
RUN bash mrireco_jl.sh
RUN rm mrireco_jl.sh
ENV PATH=/julia-1.8.3/bin:$PATH
COPY mrireco_jl_pkg.jl .
RUN julia mrireco_jl_pkg.jl
RUN rm mrireco_jl_pkg.jl

# example raw data
COPY download_data.sh .
RUN bash download_data.sh

# reconstruction code
# reconstruction code
RUN mkdir /example_code
COPY recon_scripts/run_open_source_recon.py /example_code
COPY recon_scripts/read_ismrmrd.py /example_code
COPY recon_scripts/recon_mrireco_jl_sense.jl /example_code





