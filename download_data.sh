#!/usr/bin/env bash
set -ev

# download raw data
source /opt/conda/bin/activate RecoEnv
zenodo_get 10.5281/zenodo.7468887 -o /example_data