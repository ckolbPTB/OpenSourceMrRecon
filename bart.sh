#!/usr/bin/env bash
set -ev

# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# BART
${APT_GET_INSTALL} make gcc g++ cpp libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev

wget https://github.com/mrirecon/bart/archive/v1.0.00.tar.gz
tar xzvf v1.0.00.tar.gz
cd bart-1.0.00
make CUDA=1 CUDA_BASE=/usr/local/cuda LDFLAGS="-L/usr/local/cuda/lib64"

apt-get clean


