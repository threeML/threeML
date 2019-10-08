#!/usr/bin/env bash

export TRAVIS_PYTHON_VERSION=2.7
export TRAVIS_OS_NAME=linux
export TRAVIS_BUILD_NUMBER=0
ENVNAME=test_env2_$TRAVIS_PYTHON_VERSION

# Make sure we fail in case of errors
set -e

# Environment
libgfortranver="3.0"
NUMPYVER=1.15
MATPLOTLIBVER=2
UPDATE_CONDA=true

if [[ ${TRAVIS_OS_NAME} == linux ]];
then
    miniconda_os=Linux
    compilers="gcc_linux-64 gxx_linux-64 gfortran_linux-64"
else  # osx
    miniconda_os=MacOSX
    compilers="clang_osx-64 clangxx_osx-64 gfortran_osx-64"

    # On macOS we also need the conda libx11 libraries used to build xspec
    # We also need to pin down ncurses, for now only on macos.
    xorg="xorg-libx11 ncurses=5"
fi




# Get the version in the __version__ environment variable
python ci/set_minor_version.py --patch $TRAVIS_BUILD_NUMBER --version_file threeML/version.py

export PKG_VERSION=$(cd threeML && python -c "import version;print(version.__version__)")

echo "Building ${PKG_VERSION} ..."
echo "Python version: ${TRAVIS_PYTHON_VERSION}"

if $UPDATE_CONDA ; then
    # Update conda
    echo "Update conda..."
    conda update --yes -q conda conda-build
fi

if [[ ${TRAVIS_OS_NAME} == osx ]];
then
    conda config --add channels conda-forge
fi

# Figure out requested dependencies
if [ -n "${MATPLOTLIBVER}" ]; then MATPLOTLIB="matplotlib=${MATPLOTLIBVER}"; fi
if [ -n "${NUMPYVER}" ]; then NUMPY="numpy=${NUMPYVER}"; fi

echo "dependencies: ${MATPLOTLIB} ${NUMPY}"

# Answer yes to all questions (non-interactive)
conda config --set always_yes true

# We will upload explicitly at the end, if successful
conda config --set anaconda_upload no

# Create test environment
echo "Create test environment..."
conda create --yes --name $ENVNAME -c conda-forge python=$TRAVIS_PYTHON_VERSION pytest codecov pytest-cov git ${MATPLOTLIB} ${NUMPY} astropy ${compilers}\
  libgfortran=${libgfortranver}

# Make sure conda-forge is the first channel
conda config --add channels conda-forge

conda config --add channels defaults

# Activate test environment
echo "Activate test environment..."

source $HOME/work/fermi/miniconda3/etc/profile.d/conda.sh
conda activate $ENVNAME

# Build package
echo "Build package..."
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    conda build --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
    conda index $HOME/work/fermi/miniconda3/conda-bld
else
    # there is some strange error about the prefix length
    conda build --no-build-id --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
    conda index $HOME/miniconda/conda-bld
fi
echo "======> installing..."
conda install --use-local -c conda-forge -c threeml threeml

# This is needed for ipyparallel to find the test modules
export PYTHONPATH=`pwd`/threeML/test:${PYTHONPATH}

# Make matplotlib non-interactive (otherwise it will crash
# all the tests)
export MPLBACKEND='Agg'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run tests
cd threeML/test
python -m pytest -vv --cov=threeML # -k "not slow"

# Unset PYTHONPATH and LD_LIBRARY_PATH because they conflict with anaconda client
unset PYTHONPATH
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    unset LD_LIBRARY_PATH
else
    unset DYLD_LIBRARY_PATH
fi

# Codecov needs to run in the main git repo

# Upload coverage measurements if we are on Linux
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    echo "********************************** COVERAGE ******************************"
    codecov -t 96594ad1-4ad3-4355-b177-dcb163cfc128

fi
