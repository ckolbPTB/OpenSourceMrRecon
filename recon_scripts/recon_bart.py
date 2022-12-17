#%%
import ismrmrd

import sys
#sys.path.append('/bart-0.7.00/python/')
from bart import bart
import cfl

import numpy as np

#%%
fname = '/recon_scripts/2D_cart_cine.h5'

# Read in ISMRMRD file
dset = ismrmrd.Dataset(fname, 'dataset', create_if_needed=False)

# Get header information and extract k-space dimensions
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

# Matrix size
nReadout = enc.encodedSpace.matrixSize.x
nReadout = 400
nPhaseEnc = enc.encodedSpace.matrixSize.y
nSliceEnc = enc.encodedSpace.matrixSize.z
nCoils = header.acquisitionSystemInformation.receiverChannels

print(f'k-space dimemsions: {nReadout} x {nPhaseEnc} x {nSliceEnc} x {nCoils}')

#%%
# Create k-space 
kdat = np.zeros((nReadout, nPhaseEnc, nSliceEnc, nCoils), dtype=np.complex64)

for acqnum in range(0, dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)

    if not acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
    
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        kdat[:, y, z, :] = np.moveaxis(acq.data, 0, 1)

# %%
# Write cfl file for BART
#cfl.writecfl(fname.replace('.h5', ''), kdat)
# %%
cimg = bart(1, 'fft -iu 3', kdat)

import matplotlib.pyplot as plt
plt.figure(figsize=(16,20))
for i in range(nCoils):
    plt.subplot(1, nCoils, i+1)
    plt.imshow(abs(cimg[:,:,i]).squeeze(), cmap='gray')
    plt.title('Coil image {}'.format(i))

# %%
sens = bart(1, 'ecalib -m1', kdat).squeeze()

plt.figure(figsize=(16,20))
for i in range(nCoils):
    plt.subplot(1, nCoils, i+1)
    plt.imshow(abs(sens[:,:,i]).squeeze(), cmap='gray')
    plt.title('Coil sensitivity {}'.format(i))
    
plt.figure(figsize=(16,20))
for i in range(nCoils):
    plt.subplot(1, nCoils, i+1)
    plt.imshow(abs(cimg[:,:,i]).squeeze(), cmap='gray')
    plt.title('Coil image {}'.format(i))

'''
# Combine coils by root-sum-of-squares
bart rss $(bart bitmask 3) shepp_fft shepp_fft_rss

bart pics -S -l2 -rITER shepp_usamp sens shepp_l2


ESPIRiT requires Cartesian data. For non-Cartesian data, we first reconstruct coil images using nufft, then convert them into Cartesian multi-coil k-space using fft, and then apply ecalib to estimate the sensitivities.

# Apply inverse nufft to reconstruct all eight coil images
bart nufft -i -t traj data coil_img

# Use DFT to compute Cartesian k-space
bart fft -u $(bart bitmask 0 1) coil_img ksp

# Compute sensitivities using ESPIRiT
bart ecalib -m1 ksp sens
'''
