#!/usr/bin/env bash
set -ev

# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

# update, qq: quiet
apt-get update -qq
${APT_GET_INSTALL} apt-utils locales
locale-gen en_GB.UTF-8

export LANG=en_GB.UTF-8
export LANGUAGE=en_GB:en

# base utilities
${APT_GET_INSTALL} build-essential python3-dev wget swig libomp-dev screen locate pkg-config curl git tmux zsh vim htop unzip file sshfs

# bart 1.0 requires gcc 12
#${APT_GET_INSTALL} gcc-12
#update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12

# git
${APT_GET_INSTALL} git

# ensure certificates are up to date
${APT_GET_INSTALL} --reinstall ca-certificates

# ensure nvidia signing keys are up to date
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb

apt-get clean