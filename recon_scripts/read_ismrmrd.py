#%% PACKAGES
import ismrmrd
from ismrmrdtools import transform
import numpy as np

#%% READ_ISMRMRD
def read_ismrmrd(full_fname):
    # Read in ISMRMRD file
    dset = ismrmrd.Dataset(full_fname, 'dataset', create_if_needed=False)

    # Get header information and extract k-space dimensions
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    nReadout = enc.encodedSpace.matrixSize.x
    nPixX = enc.reconSpace.matrixSize.x
    nPhaseEnc = enc.encodedSpace.matrixSize.y
    nSliceEnc = enc.encodedSpace.matrixSize.z
    nCoils = header.acquisitionSystemInformation.receiverChannels
    
    # Create k-space 
    kdat = np.zeros((nPixX, nPhaseEnc, nSliceEnc, nCoils), dtype=np.complex64)

    for acqnum in range(0, dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        if not acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):

            # Remove oversampling if needed
            if nReadout != nPixX:
                xline = transform.transform_kspace_to_image(acq.data, [1])
                x0 = (nReadout - nPixX) // 2
                x1 = (nReadout - nPixX) // 2 + nPixX
                xline = xline[:, x0:x1]
                acq.resize(nPixX, acq.active_channels, acq.trajectory_dimensions)
                acq.center_sample = nPixX//2
                # need to use the [:] notation here to fill the data
                acq.data[:] = transform.transform_image_to_kspace(xline, [1])
        
            y = acq.idx.kspace_encode_step_1
            z = acq.idx.kspace_encode_step_2
            kdat[:, y, z, :] = np.moveaxis(acq.data, 0, 1)
            
    return (kdat)

    
