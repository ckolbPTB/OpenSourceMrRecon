
using Pkg
Pkg.add("CUDA")
using MRIReco, CUDA, RegularizedLeastSquares, HDF5, NPZ, BenchmarkTools

fname = "/example_data/data.h5"
fid    = h5open(fname, "r")
kdat = read(fid["kdat"])
ktraj   = read(fid["ktraj"])
csm   = read(fid["csm"])
n_x0 = read(fid["n_x0"])
n_x1 = read(fid["n_x1"])
n_x2 = read(fid["n_x2"])
n_k0 = read(fid["n_k0"])
n_k1 = read(fid["n_k1"])
n_k2 = read(fid["n_k2"])
n_iterations = read(fid["n_iterations"])
device = read(fid["device"])
close(fid)

@show device; flush(stdout)
if device == "cuda"
    @show CUDA.functional(); flush(stdout)
    @show CUDA.device(); flush(stdout)
end

n_echoes = 1
n_slices = 1
n_repetitions = 1
n_coils = size(kdat, 2)

csm = reshape(ComplexF32.(csm), n_x0, n_x1, n_x2, n_coils)

tr = Trajectory(ktraj, n_k2*n_k1, n_k0, circular=false)

kdata = Array{Matrix{ComplexF32}}(undef, n_echoes, n_slices, n_repetitions)
kdata[1, 1, 1] = ComplexF32.(kdat)


@show size(csm); flush(stdout)     
@show size(kdat); flush(stdout)          
@show size(tr.nodes); flush(stdout)      
@show n_coils; flush(stdout)
@show n_x0; flush(stdout)
@show n_x1; flush(stdout)
@show n_x2; flush(stdout)
@show n_k0; flush(stdout)
@show n_k1; flush(stdout)
@show n_k2; flush(stdout)

@show extrema(real(tr.nodes)); flush(stdout)  # should be in [-0.5, 0.5]
@show extrema(imag(tr.nodes)); flush(stdout)

if n_x2 == 1
    encoding_size = (n_x0, n_x1)
else
    encoding_size = (n_x0, n_x1, n_x2)
end
acqData = AcquisitionData(
    Dict{Symbol,Any}(),                   
    [tr],                                 
    kdata,                                
    [collect(1:n_k0*n_k1*n_k2)],                
    encoding_size,           
    (1.0, 1.0, 1.0)                       
)

# Reconstruction
params = Dict{Symbol, Any}()
params[:reco] = "multiCoil"
params[:reconSize] = Tuple(acqData.encodingSize)
params[:iterations] = n_iterations
params[:solver] = CGNR
params[:reg] = L2Regularization(0.0)
params[:senseMaps] = csm
if device == "cuda"
    params[:arrayType] = CuArray
end

img = reconstruction(acqData, params)
@show size(img); flush(stdout) 

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 300
result = @benchmark reconstruction($acqData, $params) samples=10
println(result)

# Write reconstructed image to file
times = result.times ./ 1e9  # convert ns to seconds
npzwrite("/example_data/out.npz", Dict("img" => img, "times" => times))


