#!/usr/bin/env bash
set -ev

# download cartesian low-field data
mkdir /example_data/zip
wget "https://zenodo.org/records/19661402/files/R1_R2.zip?download=1" -O /example_data/zip/R1_R2.zip
unzip /example_data/zip/R1_R2.zip -d /example_data/zip/
mv /example_data/zip/R1_R2/noise_corr_off/9033/IR_T1w_R1.h5 /example_data/
rm -r /example_data/zip

# download radial cine data
wget "https://zenodo.org/records/15831511/files/scanner1_vol2_radial_cine.mrd?download=1" -O /example_data/scanner1_vol2_radial_cine.mrd

# download grpe data
wget "https://zenodo.org/records/20758499/files/grpe_t1_free_breathing.mrd?download=1" -O "/example_data/grpe_t1_free_breathing.mrd"
wget "https://zenodo.org/records/20758499/files/grpe_t1_free_breathing_displacement_fields.npy?download=1" -O "/example_data/grpe_t1_free_breathing_displacement_fields.npy"
