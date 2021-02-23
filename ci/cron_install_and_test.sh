#!/usr/bin/env bash

# Make sure we fail in case of errors
set -e

echo "The build number is $TRAVIS_BUILD_NUMBER"

# Testing without xspec
if (( $TRAVIS_BUILD_NUMBER % 4 == 0 )); then

    echo "Testing without xspec with python $TRAVIS_PYTHON_VERSION"

    bash install_3ML.sh --batch --python $TRAVIS_PYTHON_VERSION

# Testing with xspec
elif  (( $TRAVIS_BUILD_NUMBER % 4 == 1 )); then

    echo "Testing with xspec-modelsonly with python $TRAVIS_PYTHON_VERSION"

    bash install_3ML.sh --batch --with-xspec --python $TRAVIS_PYTHON_VERSION

# Testing with xspec and root
elif  (( $TRAVIS_BUILD_NUMBER % 4 == 2 )); then

    echo "Testing with xspec-modelsonly and root with python $TRAVIS_PYTHON_VERSION"
    bash install_3ML.sh --batch --with-xspec --with-root --python $TRAVIS_PYTHON_VERSION

# Testing with Fermi software
else

    echo "Testing with xspec-modelsonly and Fermi software with python $TRAVIS_PYTHON_VERSION"
    bash install_3ML.sh --batch --with-xspec --with-fermi --python $TRAVIS_PYTHON_VERSION

fi

source threeML_init.sh

# Now run the tests
export MPLBACKEND='Agg'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pytest -vv --pyargs threeML
pytest -vv --pyargs astromodels
