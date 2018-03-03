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

# Create test environment
conda create --name test_env -c conda-forge python=$TRAVIS_PYTHON_VERSION pytest codecov pytest-cov git

# Activate test environment
source activate test_env

# Build package

conda build -c conda-forge -c threeml --python=$TRAVIS_PYTHON_VERSION conda-dist/recipes/threeml

# Figure out where is the package
CONDA_BUILD_PATH=$(conda build conda-dist/recipes/threeml --output -c conda-forge -c threeml --python=2.7 | rev | cut -f2- -d"/" | rev)

# Install it
conda install --use-local -c conda-forge -c threeml threeml xspec-modelsonly-lite

# Run tests
cd threeml/test
python -m pytest --ignore=threeML_env -vv --cov=threeML

# Codecov needs to run in the main git repo

# Upload coverage measurements if we are on Linux
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then

    echo "********************************** COVERAGE ******************************"
    codecov -t 96594ad1-4ad3-4355-b177-dcb163cfc128

fi

# If we are on the master branch upload to the channel
if [[ "$TRAVIS_BRANCH" == "master" ]]; then

        conda install -c conda-forge anaconda-client
        anaconda -t $CONDA_UPLOAD_TOKEN upload -u threeml ${CONDA_BUILD_PATH}/*.tar.bz2 --force

else

        echo "On a branch, not uploading to Conda channel"


fi
