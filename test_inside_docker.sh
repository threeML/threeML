#!/usr/bin/env bash

# This ensure that the script will exit if any command fails
set -e

# THIS IS ASSUMED TO BE RUNNING IN THE DIRECTORY WHERE THE CODE HAS BEEN CHECKED OUT

# Setup environment
echo "##########################################################"
echo " Setting up environment"
echo "##########################################################"

source /home/hawc/.bashrc

# Test if we can import the hawc module (otherwise everything else is futile)
python -c "import hawc"

echo "##########################################################"
echo " Creating python virtual environment"
echo "##########################################################"
virtualenv threeML_env
source threeML_env/bin/activate

echo "##########################################################"
echo " Installing numpy, pytest, pytest-cov and coveralls"
echo "##########################################################"

pip install numpy pytest pytest-cov coveralls codecov

echo "##########################################################"
echo " Installing astromodels"
echo "##########################################################"

# Install the head of astromodels and the head of cthreeML, as well as py.test and coveralls
pip install git+https://github.com/giacomov/astromodels.git

echo "##########################################################"
echo " Installing 3ML"
echo "##########################################################"

# Install 3ML (current checked out version)
pip install .

echo "##########################################################"
echo " Installing cthreeML"
echo "##########################################################"

pip install git+https://github.com/giacomov/cthreeML.git

# Make the matplotlib backend non-interactive (otherwise all tests regarging plotting will fail)
export MPLBACKEND='Agg'

echo "##########################################################"
echo " Executing tests and coveralls"
echo "##########################################################"

# Execute tests
python -m pytest --ignore=threeML_env -vv --cov=threeML

echo "##########################################################"
echo " Executing coveralls"
echo "##########################################################"

# Execute the coverage analysis
coveralls

echo "##########################################################"
echo " Executing codecov"
echo "##########################################################"

# Execute the coverage analysis
codecov -t 96594ad1-4ad3-4355-b177-dcb163cfc128
