#!/usr/bin/env bash
set -ev

source /opt/conda/bin/activate reco_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/PTB-MR/mrpro.git
cd mrpro
pip install -e .