using Pkg

pkgs = ["CUDA", "NPZ", "HDF5", "BenchmarkTools"]
Pkg.add(pkgs)

Pkg.add(name="MRIReco", version="0.9.2")
Pkg.add(name="RegularizedLeastSquares", version="0.16.12")

using MRIReco, CUDA, RegularizedLeastSquares, HDF5, NPZ, BenchmarkTools