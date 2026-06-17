#!/usr/bin/env bash
set -ev

source /opt/conda/bin/activate reco_env
pip install cupy-cuda12x
pip install sigpy
