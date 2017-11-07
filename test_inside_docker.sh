#!/usr/bin/env bash

# This ensure that the script will exit if any command fails
set -e

# I am running as root, first let's create a new user and become that
# The user_id env. variable must be specified on the docker command line using -e user_id=`id -u`
adduser --system --home /home/user --shell /bin/bash --uid $user_id user --disabled-password
exec sudo -i -u user /bin/bash << EOF

set -e

cd /home/user

# Setup environment
echo "##########################################################"
echo " Setting up environment"
echo "##########################################################"

source /hawc_software/config_hawc.sh

# Need this for tests using parallel
export PYTHONPATH=/hawc_software/aerie/install/lib:/home/hawc/hawc_software/externals/2.02.02//External/root/5.34.32/lib/root:/home/hawc/hawc_software/externals/2.02.02/External/xcdf/2.10.00/lib:/home/hawc/hawc_software/externals/2.02.02//External/xcdf/2.10.00/lib:/home/hawc/hawc_software/externals/2.02.02//External/root/5.34.32/lib/root:/home/hawc/hawc_software/externals/2.02.02//External/xcdf/2.10.00/lib:/home/hawc/hawc_software/externals/2.02.02//External/root/5.34.32/lib/root:/travis_build_dir/threeML/test

# Test if we can import the hawc module (otherwise everything else is futile)
python -c "import hawc"
python -c "from hawc import liff_3ML"

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
echo " Installing ipyparallel"
echo "##########################################################"

pip install ipyparallel

echo "##########################################################"
echo " Installing astromodels"
echo "##########################################################"

# Install the head of astromodels and the head of cthreeML, as well as py.test and coveralls
pip install git+https://github.com/giacomov/astromodels.git

echo "##########################################################"
echo " Installing 3ML"
echo "##########################################################"

cd /travis_build_dir

# Install 3ML (current checked out version)
pip install . --upgrade

echo "##########################################################"
echo " Installing cthreeML"
echo "##########################################################"

pip install git+https://github.com/giacomov/cthreeML.git

echo "##########################################################"
echo " Setting up HAWC data path and try import HAWC plugin"
echo "##########################################################"

export HAWC_3ML_TEST_DATA_DIR=/hawc_test_data

# Try to import the HAWC plugin
python -c "from threeML.plugins.HAWCLike import HAWCLike"
python -c "import os; print(os.environ['HAWC_3ML_TEST_DATA_DIR'])"

# Make the matplotlib backend non-interactive (otherwise all tests regarging plotting will fail)
export MPLBACKEND='Agg'

echo "##########################################################"
echo " Executing tests and coveralls"
echo "##########################################################"

# Execute tests
python -m pytest --ignore=threeML_env -vv --cov=threeML


#echo "##########################################################"
#echo " Executing codecov"
#echo "##########################################################"
#
## Execute the coverage analysis
codecov -t 96594ad1-4ad3-4355-b177-dcb163cfc128
EOF