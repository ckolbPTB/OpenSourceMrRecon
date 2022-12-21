#%% PACKAGES
import numpy as np
import subprocess as spr
import os
import h5py

import matplotlib
font = {'size' : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

import ismrmrd
from bart import bart
import sirf.Gadgetron as pMR

from cil.optimisation.functions import LeastSquares, ZeroFunction
from cil.optimisation.algorithms import FISTA


import read_ismrmrd


#%% PARAMETERS

# Raw data file
pname = '/recon_output/simone_data_1/ChristophKolbitsch_blackBlood_raw/blackBlood/'
fname = ('bb_dat_7.h5', 'bb_dat_12.h5', 'bb_dat_15.h5',)

for scans in range(len(fname)):
        
    # List of reconstructed images and name of the method
    rec_img = []
    rec_method = []


    #%% GADGETRON (DIRECT and GRAPPA reconstruction)
    for run in range(2):
        if run == 0:
            rec_xml = 'Generic_Cartesian_FFT.xml'
        else:
            rec_xml = 'Generic_Cartesian_Spirit.xml'
            
        # Pass ISMRMRD file to gadgetron client to start image reconstruction
        gadgetron_str = 'gadgetron_ismrmrd_client -f ' + pname + fname[scans] + ' -c ' + rec_xml + ' -o ' + pname + 'out.h5'
        status = spr.run(gadgetron_str , shell=True, stdout=spr.PIPE)

        assert status.returncode == 0, 'Gadgetron reconstruction failed'

        # Read in reconstructed image data
        f = h5py.File(pname + 'out.h5')
        image_series = list(f.keys())
        img = np.array(f[image_series[0]]['image_1']['data']).squeeze()
        img = np.squeeze(img).transpose()

        # Remove reconstructed image file
        os.remove(pname + 'out.h5')

        rec_img.append(np.abs(img))
        rec_method.append('Gadgetron')


    #%% BART
    
    # Read in ISMRMRD raw data
    kdat = read_ismrmrd.read_ismrmrd(pname + fname[scans])
    
    # Calculate coil sensitivity maps
    sens = bart(1, 'ecalib -m1 -c0.001', kdat)

    # Image reconstruction
    img = bart(1, 'pics -r0 -i5', kdat, sens)

    rec_img.append(np.abs(img))
    rec_method.append('BART')


    #%% MRIRECO_JL

    # Call Julia reconstruction script
    mrireco_str = 'julia recon_mrireco_jl_sense.jl ' + pname + ' ' + fname[scans]
    status = spr.run(mrireco_str , shell=True, stdout=spr.PIPE)

    assert status.returncode == 0, 'MriReco.jl reconstruction failed'

    # Load reconstructed images
    img = np.squeeze(np.load(pname + 'out.npz'))
    
    # Remove reconstructed image file
    os.remove(pname + 'out.npz')
    
    # Remove oversampling along readout
    img = img[img.shape[0]//4:3*img.shape[0]//4,:]
    
    rec_img.append(np.abs(img))
    rec_method.append('MRIReco.jl')
        

    #%% SIRF

    # Read in ISMRMRD raw data and preprocess (remove readout oversampling)
    acq_data = pMR.AcquisitionData(pname + fname[scans])
    preprocessed_data = pMR.preprocess_acquisition_data(acq_data)

    # Calculate coil senstivity maps
    csm = pMR.CoilSensitivityData()
    csm.smoothness = 50
    csm.calculate(preprocessed_data)
    
    # Set up acquisition model
    E = pMR.AcquisitionModel(acqs=preprocessed_data, imgs=csm)
    E.set_coil_sensitivity_maps(csm)

    # Calculate the inverse
    rec_im = E.inverse(preprocessed_data)
    im_inv_uncorr = rec_im.as_array()

    # MR AcquisitionModel
    E = pMR.AcquisitionModel(acqs=preprocessed_data, imgs=rec_im)
    E.set_coil_sensitivity_maps(csm)
    
    # Starting image
    x_init = rec_im.clone()

    # Objective function
    f = LeastSquares(E, preprocessed_data, c=1)
    G = ZeroFunction()

    # Set up FISTA for least squares
    fista = FISTA(initial=x_init, f=f, g=G)
    fista.max_iteration = 10
    fista.update_objective_interval = 5

    # Run FISTA
    fista.run(10, verbose=True)

    # Get reconstructed images
    image_data = fista.get_output()
    img = image_data.as_array()
    img = np.squeeze(img).transpose()

    rec_img.append(np.abs(img))
    rec_method.append('SIRF')


    #%% VISUALISATION

    if scans == 0:
        x_y_ratio = rec_img[0].shape[0]/rec_img[0].shape[1]
        fig, ax = plt.subplots(len(fname), len(rec_img), 
                               figsize=(len(rec_img)*3, len(fname)*3*x_y_ratio), 
                               squeeze=False)
    for ind, img in enumerate(rec_img):
        img = img/np.max(img)
        if scans == 0:
            img = img[::-1, ::-1]
        if scans == 2:
            img = img[:,::-1]
        ax[scans,ind].imshow(img, cmap='gray', vmax=0.1)
        ax[scans,ind].set_xticks([])
        ax[scans,ind].set_yticks([])
        if scans == 0:
            ax[scans,ind].set_title(rec_method[ind])
plt.show()

plt.savefig('/recon_output/open_source_recon.png')