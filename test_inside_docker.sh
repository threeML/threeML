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
echo " Installing 3ML"
echo "##########################################################"

# Install 3ML (current checked out version)
pip install .

echo "##########################################################"
echo " Installing astromodels and cthreeML"
echo "##########################################################"

# Install the head of astromodels and the head of cthreeML, as well as py.test and coveralls
pip install git+https://github.com/giacomov/astromodels.git
pip install git+https://github.com/giacomov/cthreeML.git

echo "##########################################################"
echo " Installing pytest, pytest-cov and coveralls"
echo "##########################################################"

pip install pytest pytest-cov coveralls

# Make the matplotlib backend non-interactive (otherwise all tests regarging plotting will fail)
export MPLBACKEND='Agg'

echo "##########################################################"
echo " Executing tests and coveralls"
echo "##########################################################"

# Execute tests
python -m pytest -vv --cov=threeML

echo "##########################################################"
echo " Executing coveralls"
echo "##########################################################"

# Execute the coverage analysis
/home/hawc/.local/bin/coveralls