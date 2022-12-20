#!/usr/bin/env bash
set -ev

wget --quiet https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz
tar zxvf julia-1.8.3-linux-x86_64.tar.gz


git clone https://github.com/MagneticResonanceImaging/MRIReco.jl.git  --depth 1