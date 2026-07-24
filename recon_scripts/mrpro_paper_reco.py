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
    kdat = rearrange(kdata.data, '1 coils z y x -> 1 x (y z) coils')
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim 1 1 z y x -> dim x (y z)')
    return kdat.numpy(), ktraj.numpy()[::-1,...]
    
    
@contextmanager
def timer(name="", times=None, run_device='cpu'):
    if run_device == 'gpu':
        torch.cuda.synchronize()  
    start = time.perf_counter()
    yield
    if run_device == 'gpu':
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
                f"  median time={self.median_time:.4f}s (number of runs = {len(np.atleast_1d(self.times))}),\n"
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
        
    raw_files = sorted(reconstructions_dict.keys(), reverse=True)
    raw_files_str = ['3D Cartesian single-coil', '2D Radial multi-coil']
    methods = ['MRpro', 'BART', 'Sigpy', 'MriReco']
    
    fig, axes = plt.subplots(len(raw_files), len(methods), figsize=figsize)
    
    for row, (raw_file, raw_file_str) in enumerate(zip(raw_files, raw_files_str, strict=True)):
        for col, method in enumerate(methods):
            recon = reconstructions_dict[raw_file][method]
            ax = axes[row, col]
            
            vmin = np.percentile(recon.image, 1)
            vmax = np.percentile(recon.image, 99)
            
            disp_img = recon.image
            if disp_img.shape[0] > 300:
                disp_img = disp_img[80:240, 80:240]
                disp_img = np.rot90(disp_img, -1)
            else:
                disp_img = np.rot90(disp_img, 1)
            
            # Plot with consistent scaling
            ax.imshow(disp_img, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Top: method name (first row only)
            if row == 0:
                ax.set_title(method, fontsize=20, fontweight='bold')
            
            # Left: raw file name (first column only)
            if col == 0:
                ax.set_ylabel(raw_file_str, fontsize=18, fontweight='bold')
            
            # Info box
            if method != 'MRpro':
                ax.text(0.98, 0.98, f"Diff to MRpro: {recon.relative_rmse*100:.1f}%",
                        transform=ax.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=14)
            
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

run_device = 'cpu'

# RESULTS
reconstruction_results = []

for fname in [pname + 'cart_Ruben_R1_R2_9033_IR_T1w_R1.h5', pname + '20250210-154640,PTBMR06-73,20250210_radial_2D_cine_640mm_3600spokes_heart_rate65,13633,226,dep13622_with_traj.mrd']:

    # PREP
    if 'cart' in fname:
        kdata = KData.from_file(fname, KTrajectoryCartesian())
        #kdata = kdata.remove_readout_os()
        n_iterations = 1
        n_iterations_mrireco = 1
        nruns = 10
        ndim_traj = 3
        disp_x = 25
    else:
        kdata = KData.from_file(fname, KTrajectoryIsmrmrd())
        kdata = kdata[...,400:400+640,:]
        kdata.data *= 100
        n_iterations = 20
        n_iterations_mrireco = 5
        nruns = 10
        ndim_traj = 2
        disp_x = 0
        
        
    mask = 1 #circ_mask()
    csm = CsmData.from_kdata_inati(kdata[0])

    # BART
    kdat_bart, ktraj_bart = kdata_ktraj_to_bart(kdata)
    csm_bart = rearrange(csm.data, '1 coils z y x -> x y z coils').numpy()

    # SIGPY
    kdat_sigpy = rearrange(kdat_bart, '1 x yz coils -> coils (x yz)')
    if ndim_traj == 3:
        csm_sigpy = rearrange(csm_bart, 'x y z coils -> coils x y z')
    else:
        csm_sigpy = rearrange(csm_bart, 'x y 1 coils -> coils x y')
    ktraj_sigpy = rearrange(np.copy(ktraj_bart[:ndim_traj,...]), 'dim x yz-> (x yz) dim')

    # MRIRECO
    kdat_mrireco = rearrange(kdat_bart, '1 x yz coils -> coils (x yz)')
    csm_mrireco = rearrange(csm_bart, 'x y z coils -> coils z y x')
    ktraj_mrireco = rearrange(np.copy(ktraj_bart[:ndim_traj,...]), 'dim x yz -> (x yz) dim')
    ktraj_mrireco[...,0] = ktraj_mrireco[...,0]/kdata.header.encoding_matrix.x
    ktraj_mrireco[...,1] = ktraj_mrireco[...,1]/kdata.header.encoding_matrix.y
    if ndim_traj == 3:
        ktraj_mrireco[...,2] = ktraj_mrireco[...,2]/kdata.header.encoding_matrix.z

    with h5py.File(pname + 'data.h5', 'w') as f:
        f.create_dataset('kdat', data=kdat_mrireco)
        f.create_dataset('ktraj',   data=ktraj_mrireco)
        f.create_dataset('csm',   data=csm_mrireco)
        f.create_dataset('n_k0', data=kdata.shape[-1])
        f.create_dataset('n_k1', data=kdata.shape[-2])
        f.create_dataset('n_k2', data=kdata.shape[-3])
        f.create_dataset('n_x0', data=kdata.header.recon_matrix.x)
        f.create_dataset('n_x1', data=kdata.header.recon_matrix.y)
        f.create_dataset('n_x2', data=kdata.header.recon_matrix.z)
        f.create_dataset('n_iterations', data=n_iterations_mrireco)
        f.create_dataset('device', data=run_device)
        
        


    # MRPRO
    recon = IterativeSENSEReconstruction(kdata, csm=csm, n_iterations=n_iterations)
    if run_device == 'gpu':
        recon.cuda()
        kdata = kdata.cuda()

    times = []
    for _ in range(nruns):
        with timer("mrpro", times=times, run_device=run_device):     
            idata = recon(kdata)
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    img_mrpro = idata.data.cpu().squeeze().abs().numpy()
    img_mrpro = img_mrpro[None] if img_mrpro.ndim == 2 else img_mrpro
    img_mrpro /= np.max(img_mrpro)
    plt.figure()
    plt.imshow(img_mrpro[disp_x])
    plt.savefig(pname + 'mrpro.png', dpi=300)
    
    reconstruction_results.append(Reconstruction(img_mrpro[disp_x], relative_rmse(img_mrpro, img_mrpro), times, 'MRpro', fname, run_device == 'gpu'))
    
    # MRIRECO_JL
    mrireco_str = 'julia /code/mrpro_paper_julia.jl ' + fname
    status = spr.run(mrireco_str , shell=True)
    assert status.returncode == 0, 'MriReco.jl reconstruction failed'
    print(np.load(pname + 'out.npz'))
    img = np.abs(np.squeeze(np.load(pname + 'out.npz')['img']))
    img = img[...,None] if img.ndim == 2 else img
    img = rearrange(img, 'x y z -> z y x')
    img /= np.max(img)
    times = np.squeeze(np.load(pname + 'out.npz')['times'])
    os.remove(pname + 'out.npz')
    
    plt.figure()
    plt.imshow(img[disp_x])
    plt.savefig(pname + 'mrireco.png', dpi=300)

    reconstruction_results.append(Reconstruction(img[disp_x], relative_rmse(img_mrpro*mask, img), times, 'MriReco', fname, run_device == 'gpu'))


    print(f'bart {np.max(ktraj_bart)}')

     # BART
    times = []
    for _ in range(nruns):
        if run_device == 'gpu':
            with timer("bart", times=times, run_device=run_device): 
                img = bart(1, f'pics -r0 -g -i{n_iterations} -t', ktraj_bart, kdat_bart, csm_bart)
        else:
            with timer("bart", times=times, run_device=run_device): 
                img = bart(1, f'pics -r0 -i{n_iterations} -t', ktraj_bart, kdat_bart, csm_bart)
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    img = np.squeeze(np.abs(img))
    img = img[...,None] if img.ndim == 2 else img
    img = rearrange(img, 'x y z -> z y x')
    img /= np.max(img)
    plt.figure()
    plt.imshow(img[disp_x])
    plt.savefig(pname + 'bart.png', dpi=300)

    reconstruction_results.append(Reconstruction(img[disp_x], relative_rmse(img_mrpro, img), times, 'BART', fname, run_device == 'gpu'))


    print(f'sigpy {np.max(ktraj_sigpy)}')

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
        with timer("sigpy", times=times, run_device=run_device):     
            img = recon.run()
    times = np.array(times)
    print(f"median: {np.median(times):.3f}s | min: {times.min():.3f}s | max: {times.max():.3f}s\n\n")

    if run_device == 'gpu':
        img = img.get()
    img = np.squeeze(np.abs(img))
    img = img[...,None] if img.ndim == 2 else img
    img = rearrange(img, 'x y z -> z y x')
    img /= np.max(img)
    plt.figure()
    plt.imshow(img[disp_x])
    plt.savefig(pname + 'sigpy.png', dpi=300)

    reconstruction_results.append(Reconstruction(img[disp_x], relative_rmse(img_mrpro, img), times, 'Sigpy', fname, run_device == 'gpu'))

for recon in reconstruction_results:
    print(recon)

for recon in reconstruction_results:
    print(f'{recon.method}: {recon.median_time:.4f}s (cuda = {recon.cuda})')
    
fig = plot_reconstructions(reconstruction_results)
plt.savefig('/data/compare_to_packages.png', dpi=300, bbox_inches='tight')
plt.show()

