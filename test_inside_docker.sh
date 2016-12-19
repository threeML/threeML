#!/usr/bin/env bash

# THIS IS ASSUMED TO BE RUNNING IN THE DIRECTORY WHERE THE CODE HAS BEEN CHECKED OUT

# Setup environment
source ~/.bashrc

# Test if we can import the hawc module (otherwise everything else is futile)
python -c "import hawc"

# Install 3ML (current checked out version)
pip install .

# Install the head of astromodels and the head of cthreeML, as well as py.test and coveralls
pip install git+https://github.com/giacomov/astromodels.git
pip install git+https://github.com/giacomov/cthreeML.git
pip install pytest pytest-cov coveralls

# Make the matplotlib backend non-interactive (otherwise all tests regarging plotting will fail)
export MPLBACKEND='Agg'

# Execute tests
python -m pytest -vv --cov=threeML

# Execute the coverage analysis
/home/hawc/.local/bin/coveralls