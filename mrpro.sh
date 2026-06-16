#!/usr/bin/env bash
set -ev

git clone https://github.com/PTB-MR/mrpro.git
cd mrpro
pip install -e ".[dev]"