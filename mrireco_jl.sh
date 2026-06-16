#!/usr/bin/env bash
set -ev

wget --quiet https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz
tar zxvf julia-1.8.3-linux-x86_64.tar.gz
mv julia-1.8.3 /opt/julia
ln -s /opt/julia/bin/julia /usr/local/bin/julia
rm julia-1.8.3-linux-x86_64.tar.gz

git clone https://github.com/MagneticResonanceImaging/MRIReco.jl.git  --depth 1