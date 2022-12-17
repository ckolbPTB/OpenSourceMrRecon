#!/usr/bin/env bash
set -ev

# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# BART
${APT_GET_INSTALL} make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev

wget https://github.com/mrirecon/bart/archive/v0.7.00.tar.gz
tar xzvf v0.7.00.tar.gz
cd bart-0.7.00
make 


