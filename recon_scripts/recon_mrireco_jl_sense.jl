using Pkg

Pkg.add("NPZ")
using NPZ

Pkg.add("MRIReco")
using MRIReco
isinstalled(pkg::String) = any(x -> x.name == pkg && x.is_direct_dep, values(Pkg.dependencies()))

# Install required packages
for P in ["HTTP", "PyPlot", "MRIReco", "MRIFiles", "MRICoilSensitivities"]
  !isinstalled(P) && Pkg.add(P)
end

using MRIReco, MRIFiles, MRICoilSensitivities

# Load ISMRMRD raw data
fname = ARGS[1] * ARGS[2]
f = ISMRMRDFile(fname)
acqRaw = RawAcquisitionData(f)
acqData = AcquisitionData(acqRaw; estimateProfileCenter=true)

smaps = espirit(acqData, (6,6), 30, eigThresh_1=0.001, eigThresh_2=0.98)

# Reconstruction
params = Dict{Symbol, Any}()
params[:reco] = "multiCoil"
params[:reconSize] = Tuple(acqData.encodingSize[1:2])
params[:regularization] = "L2"
params[:iterations] = 5
params[:solver] = "cgnr"
params[:senseMaps] = smaps
img = reconstruction(acqData, params)

# Write reconstructed image to file
npzwrite(ARGS[1] * "out.npz", img)

