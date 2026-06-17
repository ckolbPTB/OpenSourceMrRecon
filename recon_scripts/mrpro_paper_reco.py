# %%
import matplotlib.pyplot as plt
from dataclasses import dataclass
import subprocess as spr
import os
import numpy as np
import h5py
import torch

from contextlib import contextmanager
import time

import sigpy.mri as mr
import sigpy as sp

import ismrmrd
from ismrmrdtools import transform

from einops import rearrange

from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction
from mrpro.data.traj_calculators import KTrajectoryCartesian, KTrajectoryIsmrmrd
from mrpro.data import KData, CsmData
from mrpro.operators import CartesianSamplingOp
from bart import bart



def kdata_ktraj_to_bart(kdata):
    kdat = rearrange(kdata.data, 'other coils z y x -> 1 x y coils z other')
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim other coils z y x -> dim x y z coils other')
    return kdat.numpy(), ktraj.numpy()[::-1,...]
    
    
@contextmanager
def timer(name="", times=None):
    torch.cuda.synchronize()  
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if times is not None:
        times.append(elapsed)
    else:
        print(f"{name}: {elapsed:.3f}s")
        
        
def relative_rmse(img1, img2):
    mse = np.mean((img1 - img2)**2)
    norm = np.mean(np.abs(img1)**2)
    return np.sqrt(mse / norm)
        
        
@dataclass
class Reconstruction:
    image: np.ndarray
    relative_rmse: float
    times: np.ndarray
    method: str
    raw_file: str
    cuda: bool
    
    @property
    def median_time(self) -> float:
        return np.median(self.times)
    
    def __repr__(self):
        return (f"{self.method} ({self.raw_file})\n"
                f"  cuda={self.cuda},\n"
                f"  median time={self.median_time:.4f}s (number of runs = {len(self.times)}),\n"
                f"  relative RMSE={self.relative_rmse:.4f} (image shape = {self.image.shape})\n"
                f")")
        
        
def plot_reconstructions(reconstructions_list, vmin=None, vmax=None, figsize=(16, 8)):
    """Publication-quality reconstruction comparison."""
    
    # Group by raw_file and method
    reconstructions_dict = {}
    for recon in reconstructions_list:
        if recon.raw_file not in reconstructions_dict:
            reconstructions_dict[recon.raw_file] = {}
        reconstructions_dict[recon.raw_file][recon.method] = recon
        
    raw_files = sorted(reconstructions_dict.keys())
    raw_files_str = ['Cartesian', 'Golden Radial']
    methods = ['MRpro', 'Bart', 'Sigpy', 'MriReco']
    
    fig, axes = plt.subplots(len(raw_files), len(methods), figsize=figsize)
    
    for row, (raw_file, raw_file_str) in enumerate(zip(raw_files, raw_files_str, strict=True)):
        for col, method in enumerate(methods):
            recon = reconstructions_dict[raw_file][method]
            ax = axes[row, col]
            
            vmin = np.percentile(recon.image, 1)
            vmax = np.percentile(recon.image, 99)
            
            # Plot with consistent scaling
            ax.imshow(recon.image, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Top: method name (first row only)
            if row == 0:
                ax.set_title(method, fontsize=12, fontweight='bold')
            
            # Left: raw file name (first column only)
            if col == 0:
                ax.set_ylabel(raw_file_str, fontsize=11, fontweight='bold')
            
            # Info box
            ax.text(0.98, 0.98, f"RMSE: {recon.relative_rmse*100:.1f}%\nTime: {recon.median_time:.3f}s",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def circ_mask(size=256, radius=128):
    # Create circular mask
    center = size // 2

    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    mask = distance <= radius
    return mask

# %%
pname = '/data/'

run_device = 'gpu'

# RESULTS
reconstruction_results = []

for fname in [pname + 'cart_t1.mrd', pname + 'radial2D_96spokes_golden_angle_with_traj.h5']:

    # PREP
    if 'cart' in fname:
        kdata = KData.from_file(fname, KTrajectoryCartesian())
        kdata = kdata.remove_readout_os()
        n_iterations = 1
        nruns = 15
        mask = 1
    else:
        kdata = KData.from_file(fname, KTrajectoryIsmrmrd())
        n_iterations = 20
        nruns = 6
        mask = circ_mask()
        
    csm = CsmData.from_kdata_inati(kdata[0])

    # BART
    kdat_bart, ktraj_bart = kdata_ktraj_to_bart(kdata)
    csm_bart = rearrange(csm.data, 'other coils z y x -> x y z coils other').numpy()

    # SIGPY
    kdat_sigpy = rearrange(np.squeeze(kdat_bart), 'x y coils -> coils (x y)')
    csm_sigpy = rearrange(np.squeeze(csm_bart), 'x y coils -> coils x y')
    ktraj_sigpy = rearrange(np.squeeze(ktraj_bart[:2,...]), 'dim x y -> (x y) dim')

    # MRIRECO
    kdat_mrireco = rearrange(np.squeeze(kdat_bart), 'x y coils -> coils (x y)')
    csm_mrireco = rearrange(np.squeeze(csm_bart), 'x y coils -> coils y x')
    ktraj_mrireco = rearrange(np.squeeze(ktraj_bart[:2,...]), 'dim x y -> (x y) dim')
    ktraj_mrireco = ktraj_mrireco/kdata.header.recon_matrix.x

    with h5py.File(pname + 'data.h5', 'w') as f:
        f.create_dataset('kdat', data=kdat_mrireco)
        f.create_dataset('ktraj',   data=ktraj_mrireco)
        f.create_dataset('csm',   data=csm_mrireco)
        f.create_dataset('matrix_size', data=kdata.header.recon_matrix.x)
        f.create_dataset('n_k0', data=kdata.shape[-1])
        f.create_dataset('n_k1', data=kdata.shape[-2])
        f.create_dataset('n_iterations', data=n_iterations)
        f.create_dataset('device', data=run_device)
        
        


    # MRPRO
    recon = IterativeSENSEReconstruction(kdata, csm=csm, n_iterations=n_iterations)
    if run_device == 'gpu':
        recon.cuda()
        kdata = kdata.cuda()

    times = []
    for _ in range(nruns):
        with timer("mrpro", times=times):     
            idata = recon(kdata)
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    img_mrpro = idata.data.cpu().squeeze().abs().numpy()
    img_mrpro /= np.max(img_mrpro)
    plt.figure()
    plt.imshow(img_mrpro)
    plt.savefig(pname + 'mrpro.png', dpi=300)

    reconstruction_results.append(Reconstruction(img_mrpro, relative_rmse(img_mrpro, img_mrpro), times, 'MRpro', fname, run_device == 'gpu'))



    # MRIRECO_JL
    mrireco_str = 'julia /code/mrpro_paper_julia.jl ' + fname
    status = spr.run(mrireco_str , shell=True)
    assert status.returncode == 0, 'MriReco.jl reconstruction failed'
    print(np.load(pname + 'out.npz'))
    img = np.abs(np.squeeze(np.load(pname + 'out.npz')['img']))
    img = rearrange(img, 'x y -> y x')
    img /= np.max(img)
    times = np.squeeze(np.load(pname + 'out.npz')['times'])
    os.remove(pname + 'out.npz')

    plt.figure()
    plt.imshow(img)
    plt.savefig(pname + 'mrireco.png', dpi=300)

    reconstruction_results.append(Reconstruction(img, relative_rmse(img_mrpro*mask, img), times, 'MriReco', fname, run_device == 'gpu'))



    # SIGPY
    device_id = 0 if run_device == 'gpu' else -1
    times = []
    for _ in range(nruns):
        recon = mr.app.SenseRecon(
            sp.to_device(kdat_sigpy, device_id),
            sp.to_device(csm_sigpy, device_id),
            lamda=0.0,
            coord=sp.to_device(ktraj_sigpy, device_id),
            max_iter=n_iterations,
            device=device_id,
        )
        with timer("sigpy", times=times):     
            img = recon.run()
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    img = np.squeeze(np.abs(img.get()))
    img = rearrange(img, 'x y -> y x')
    img /= np.max(img)
    plt.figure()
    plt.imshow(img)
    plt.savefig(pname + 'sigpy.png', dpi=300)

    reconstruction_results.append(Reconstruction(img, relative_rmse(img_mrpro, img), times, 'Sigpy', fname, run_device == 'gpu'))



    # BART
    times = []
    for _ in range(nruns):
        if run_device == 'gpu':
            with timer("bart", times=times): 
                img = bart(1, f'pics -r0 -g -i{n_iterations} -t', ktraj_bart, kdat_bart, csm_bart)
        else:
            with timer("bart", times=times): 
                img = bart(1, f'pics -r0 -i{n_iterations} -t', ktraj_bart, kdat_bart, csm_bart)
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    img = np.squeeze(np.abs(img))
    img = rearrange(img, 'x y -> y x')
    img /= np.max(img)
    plt.figure()
    plt.imshow(img)
    plt.savefig(pname + 'bart.png', dpi=300)

    reconstruction_results.append(Reconstruction(img, relative_rmse(img_mrpro, img), times, 'Bart', fname, run_device == 'gpu'))


for recon in reconstruction_results:
    print(recon)
    
fig = plot_reconstructions(reconstruction_results)
plt.savefig('/data/compare_to_packages.png', dpi=300, bbox_inches='tight')
plt.show()
    

