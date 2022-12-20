using Pkg

Pkg.add("MRIReco")
using MRIReco
isinstalled(pkg::String) = any(x -> x.name == pkg && x.is_direct_dep, values(Pkg.dependencies()))

# Install required packages
for P in ["HTTP", "PyPlot", "NPZ", "MRIReco", "MRIFiles", "MRICoilSensitivities"]
  !isinstalled(P) && Pkg.add(P)
end

using NPZ
using MRIReco, MRIFiles, MRICoilSensitivities