#!/usr/bin/env bash
# Make sure we fail in case of errors
set -e

TRAVIS_OS_NAME="unknown"

if [[ "$OSTYPE" == "linux-gnu" ]]; then

        # Linux

        TRAVIS_OS_NAME="linux"


elif [[ "$OSTYPE" == darwin* ]]; then

        # Mac OSX

        TRAVIS_OS_NAME="osx"


elif [[ "$OSTYPE" == "cygwin" ]]; then

        # POSIX compatibility layer and Linux environment emulation for Windows

        TRAVIS_OS_NAME="linux"

else

        # Unknown.

        echo "Could not guess your OS. Exiting."

        exit 1

fi

echo "Running on ${TRAVIS_OS_NAME}"

TRAVIS_PYTHON_VERSION=2.7
export TRAVIS_BUILD_NUMBER=2
ENVNAME=threeML_test_$TRAVIS_PYTHON_VERSION
USE_LOCAL=false

# Environment
libgfortranver="3.0"
NUMPYVER=1.15
MATPLOTLIBVER=2
UPDATE_CONDA=false
XSPECVER="6.22.1"
xspec_channel=threeml

if [[ ${TRAVIS_OS_NAME} == "linux" ]];
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
echo "Use local is: ${USE_LOCAL}"

if ${USE_LOCAL}; then
    conda config --remove channels ${xspec_channel}
    use_local="--use-local"
else
    conda config --add channels ${xspec_channel}
fi

if $UPDATE_CONDA ; then
    # Update conda
    echo "Update conda..."
    conda update --yes -q conda conda-build
fi

# Figure out requested dependencies
if [ -n "${MATPLOTLIBVER}" ]; then MATPLOTLIB="matplotlib=${MATPLOTLIBVER}"; fi
if [ -n "${NUMPYVER}" ]; then NUMPY="numpy=${NUMPYVER}"; fi
if [ -n "${XSPECVER}" ];
 then export XSPEC="xspec-modelsonly=${XSPECVER} ${xorg}";
fi

echo "dependencies: ${MATPLOTLIB} ${NUMPY} ${XSPEC}"

# Answer yes to all questions (non-interactive)
conda config --set always_yes true

# We will upload explicitly at the end, if successful
conda config --set anaconda_upload no

# Make sure conda-forge is the first channel
conda config --add channels conda-forge/label/cf201901

conda config --add channels conda-forge

conda config --add channels defaults

conda config --add channels threeml

# Create test environment
echo "Create test environment..."
conda create --yes --name $ENVNAME -c conda-forge ${use_local} python=$TRAVIS_PYTHON_VERSION "pytest<4" codecov pytest-cov git ${MATPLOTLIB} ${NUMPY} ${XSPEC} astropy ${compilers} scipy openblas-devel=0.3.6 tk=8.5.19
#libgfortran=${libgfortranver}
#openblas-devel=0.3.6 openblas=0.2.20 blas=1.1

# Activate test environment
echo "Activate test environment..."

source $CONDA_PREFIX/etc/profile.d/conda.sh
#source $HOME/work/fermi/miniconda3/etc/profile.d/conda.sh
conda activate $ENVNAME

# Build package
echo "Build package..."
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    conda build --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
else
    # there is some strange error about the prefix length
    conda build --no-build-id --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
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
