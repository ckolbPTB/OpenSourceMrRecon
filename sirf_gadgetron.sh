#!/usr/bin/env bash
set -ev

source /opt/conda/bin/activate RecoEnv

# define apt-get installation command
APT_GET_INSTALL="apt-get install -yq --no-install-recommends"

${APT_GET_INSTALL} libhdf5-serial-dev libboost-dev libboost-all-dev libfftw3-dev h5utils  jq  \
  hdf5-tools libopenblas-dev libxml2-dev  libfreetype6-dev       \
  libxslt-dev            \
  libarmadillo-dev       \
  libace-dev             \
  liblapack-dev          \
  liblapacke-dev         \
  libplplot-dev          \
  libdcmtk-dev           \
  libpugixml-dev         \
  libgflags-dev          \
  libssl-dev             \
  libcurl4-openssl-dev   \
  golang

  apt-get clean

# SIRF-SuperBuild 
SIRF_SB_URL="https://github.com/SyneRBI/SIRF-SuperBuild"
SIRF_SB_TAG="master"
NUM_PARALLEL_BUILDS="20"
INSTALL_DIR="/opt"

BUILD_FLAGS="\
 -DCMAKE_BUILD_TYPE=Release\
 -DSTIR_ENABLE_OPENMP=ON -DUSE_SYSTEM_ACE=ON\
 -DUSE_SYSTEM_Armadillo=ON -DUSE_SYSTEM_Boost=ON\
 -DUSE_SYSTEM_FFTW3=ON -DUSE_SYSTEM_HDF5=ON -DUSE_ITK=OFF\
 -DGadgetron_USE_CUDA=OFF\
 -DUSE_SYSTEM_SWIG=ON\
 -DUSE_NiftyPET=OFF\
 -DBUILD_siemens_to_ismrmrd=OFF -DBUILD_pet_rd_tools=OFF"
EXTRA_BUILD_FLAGS="-DBUILD_CIL=ON"

git clone "$SIRF_SB_URL" --recursive "$INSTALL_DIR"/SIRF-SuperBuild
cd $INSTALL_DIR/SIRF-SuperBuild
git checkout "$SIRF_SB_TAG"

COMPILER_FLAGS="-DCMAKE_C_COMPILER='$(which gcc)' -DCMAKE_CXX_COMPILER='$(which g++)'"
g++ --version
cmake --version
echo "PATH: $PATH"
echo "COMPILER_FLAGS: $COMPILER_FLAGS"
echo "BUILD+EXTRA FLAGS: $BUILD_FLAGS $EXTRA_BUILD_FLAGS"

cmake $BUILD_FLAGS $EXTRA_BUILD_FLAGS $COMPILER_FLAGS .

cmake --build . -j ${NUM_PARALLEL_BUILDS}