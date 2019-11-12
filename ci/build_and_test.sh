#!/usr/bin/env bash

# Make sure we fail in case of errors
set -e

# Copy sources (we do not have write permission on the mounted $TRAVIS_BUILD_DIR),
# so let's make a copy of the source code
cd ~
rm -rf my_work_dir
mkdir my_work_dir
# Copy also dot files (.*)
shopt -s dotglob
cp -R ${TRAVIS_BUILD_DIR}/* my_work_dir/

cd my_work_dir

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

echo "HOME= ${HOME}"
echo "Building ${PKG_VERSION} ..."
echo "Python version: ${TRAVIS_PYTHON_VERSION}"

libgfortranver="3.0"
NUMPYVER=1.15
MATPLOTLIBVER=2
XSPECVER="6.22.1"
xspec_channel=threeml

echo "Building ${PKG_VERSION} ..."
echo "Python version: ${TRAVIS_PYTHON_VERSION}"

conda update --yes -q conda conda-build

conda config --add channels ${xspec_channel}

# Figure out requested dependencies
if [ -n "${MATPLOTLIBVER}" ]; then MATPLOTLIB="matplotlib=${MATPLOTLIBVER}"; fi
if [ -n "${NUMPYVER}" ]; then NUMPY="numpy=${NUMPYVER}"; fi
if [ -n "${XSPECVER}" ];
 then export XSPEC="xspec-modelsonly=${XSPECVER} ${xorg}";
fi
echo "dependencies: ${MATPLOTLIB} ${NUMPY}  ${XSPEC}"

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
conda create --yes --name test_env -c conda-forge python=$TRAVIS_PYTHON_VERSION "pytest<4" codecov pytest-cov git ${MATPLOTLIB} ${NUMPY} ${XSPEC} astropy ${compilers} scipy openblas-devel=0.3.6 tk=8.5.19

if [[ "$TRAVIS_OS_NAME" == "removeme" ]]; then

    # in the hawc docker container we have a configuration file

    source ${SOFTWARE_BASE}/config_hawc.sh
    source activate test_env
    conda install -c conda-forge pytest=3.8 codecov pytest-cov git --no-update-deps
else

    # Activate test environment
    source activate test_env

fi

# Build package

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    conda build --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
else
    # there is some strange error about the prefix length
    conda build --no-build-id --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml
fi

# Install it
conda install --use-local -c conda-forge -c threeml threeml

########### FIXME
# This is a kludge around a pymultinest bug
# (it cannot find multinest if not in LD_LIBRARY_PATH
# or DYLD_LIBRARY_PATH)
#if [[ "$TRAVIS_OS_NAME" == "removeme" ]]; then
#    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
#else
#    export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
#fi

# Run tests
cd ~/my_work_dir/

# This is needed for ipyparallel to find the test modules
export PYTHONPATH=`pwd`/threeML/test:${PYTHONPATH}


# Make matplotlib non-interactive (otherwise it will crash
# all the tests)
export MPLBACKEND='Agg'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Before running the test, if we are on linux, install cthreeml and verify that
# we can actually import the HAWC plugin
# We re-install cthreeML to make sure that it uses versions of boost compatible
# with what is installed in the container
if [[ "$TRAVIS_OS_NAME" == "removeme" ]]; then

    export CFLAGS="-m64 -I${CONDA_PREFIX}/include"
    export CXXFLAGS="-DBOOST_MATH_DISABLE_FLOAT128 -m64 -I${CONDA_PREFIX}/include"
    pip install git+https://github.com/threeml/cthreeML.git --no-deps --upgrade

    # Make sure we can load the HAWC plugin
    python -c "from threeML.plugins.HAWCLike import HAWCLike"
    python -c "import os; print(os.environ['HAWC_3ML_TEST_DATA_DIR'])"

fi

python -m pytest -vv --cov=threeML

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

# If we are on the master branch upload to the channel
if [[ "${TRAVIS_EVENT_TYPE}" == "pull_request" ]]; then

    echo "This is a pull request, not uploading to Conda channel"

else
    if [[ "${TRAVIS_EVENT_TYPE}" == "push" ]]; then

        echo "This is a push to TRAVIS_BRANCH=${TRAVIS_BRANCH}"

        if [[ "${TRAVIS_BRANCH}" == "master" ]]; then

            conda install -c anaconda-client

            echo "Uploading ${CONDA_BUILD_PATH}"

            if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

                    anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml ${HOME}/miniconda/conda-bld/linux-64/*.tar.bz2 --force

            else

                    anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml ${HOME}/miniconda/conda-bld/*/*.tar.bz2 --force
            fi
        fi
    fi
    if [[ "${TRAVIS_EVENT_TYPE}" == "removeme" ]]; then

        echo "This is a push, uploading to Conda channel"

        source activate root

        conda install anaconda-client
            
        if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
                
            anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml ${HOME}/conda-bld/linux-64/*.tar.bz2 --force
            
        else
            
            anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml ${HOME}/miniconda/conda-bld/osx-64/*.tar.bz2 --force
            
        fi
    fi
fi
