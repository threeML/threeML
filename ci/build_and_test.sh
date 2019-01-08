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

# Get the version in the __version__ environment variable
python ci/set_minor_version.py --patch $TRAVIS_BUILD_NUMBER --version_file threeML/version.py

export PKG_VERSION=$(cd threeML && python -c "import version;print(version.__version__)")

echo "Building ${PKG_VERSION} ..."

# Update conda
conda update --yes -q conda #conda-build

# Answer yes to all questions (non-interactive)
conda config --set always_yes true

# We will upload explicitly at the end, if successful
conda config --set anaconda_upload no

# Make sure conda-forge is the first channel
conda config --add channels conda-forge


if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    # in the hawc docker container we have a configuration file

    source ${SOFTWARE_BASE}/config_hawc.sh
    source activate test_env
    conda install -c conda-forge pytest=3.10 codecov pytest-cov git --no-update-deps
else

    # Activate test environment
    source activate test_env

fi

# Build package
cd conda-dist/recipes/threeml
conda build -c conda-forge -c threeml --python=$TRAVIS_PYTHON_VERSION .

# Install it
conda install --use-local -c threeml -c conda-forge pygmo=2.4 threeml xspec-modelsonly-lite

########### FIXME
# This is a kludge around a pymultinest bug
# (it cannot find multinest if not in LD_LIBRARY_PATH
# or DYLD_LIBRARY_PATH)
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

else

    export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

fi

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
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    export CFLAGS="-m64 -I${CONDA_PREFIX}/include"
    export CXXFLAGS="-DBOOST_MATH_DISABLE_FLOAT128 -m64 -I${CONDA_PREFIX}/include"
    pip install git+https://github.com/giacomov/cthreeML.git --no-deps --upgrade

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

            echo "This is a push, uploading to Conda channel"

            source activate root

            conda install anaconda-client
            
            if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
                
                anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml /opt/conda/conda-bld/linux-64/*.tar.bz2 --force
            
            else
            
                anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml /Users/travis/miniconda/conda-bld/osx-64/*.tar.bz2 --force
            
            fi
        fi
fi
