#!/usr/bin/env bash

# Make sure we fail in case of errors
set -e

echo "The build number is $TRAVIS_BUILD_NUMBER"

# Testing without xspec
if (( $TRAVIS_BUILD_NUMBER % 4 == 0 )); then

    echo "Testing without xspec"

    bash install_3ML.sh --batch

# Testing with xspec
elif  (( $TRAVIS_BUILD_NUMBER % 4 == 1 )); then

    echo "Testing with xspec-modelsonly"

    bash install_3ML.sh --batch --with-xspec

# Testing with xspec and root
elif  (( $TRAVIS_BUILD_NUMBER % 4 == 2 )); then

    echo "Testing with xspec-modelsonly and root"

    bash install_3ML.sh --batch --with-xspec --with-root

# Testing with Fermi software
else

    echo "Testing with xspec-modelsonly and Fermi software"

    bash install_3ML.sh --batch --with-xspec --with-fermi

fi

source threeML_init.sh

# Now run the tests
export MPLBACKEND='Agg'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pytest -vv --pyargs threeML
pytest -vv --pyargs astromodels
