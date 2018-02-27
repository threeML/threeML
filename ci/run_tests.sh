#!/usr/bin/env bash

set -e

# Setup environment
echo "##########################################################"
echo " Setting up environment"
echo "##########################################################"

source /hawc_software/config_hawc.sh

# Need this for tests using parallel
export PYTHONPATH=/travis_build_dir/threeML/test:/hawc_software/trunk/install/lib:/opt/conda/envs/test_env/lib/root:/hawc_software/externals/2.06.00/External/xcdf/3.00.01/lib:/opt/rh/devtoolset-2/root/usr/lib64/python/site-packages

# Test if we can import the hawc module (otherwise everything else is futile)
python -c "import hawc"
python -c "from hawc import liff_3ML"

echo "##########################################################"
echo " Setting up test environment"
echo "##########################################################"
source activate test_env

conda install -y pytest pytest-cov coveralls codecov \
                 ipyparallel astropy numdifftools pandas \
                 pytables zlib=1.2.8 dill emcee astroquery \
                 uncertainties pyyaml iminuit corner \
                 requests speclite ipython boost=1.63 \
                 pymultinest pygmo

echo "##########################################################"
echo " Installing astromodels"
echo "##########################################################"

export CFLAGS="-m64 -I${CONDA_PREFIX}/include"
export CXXFLAGS="-DBOOST_MATH_DISABLE_FLOAT128 -m64 -I${CONDA_PREFIX}/include"

pip install git+https://github.com/giacomov/astromodels.git --no-deps

echo "##########################################################"
echo " Installing 3ML"
echo "##########################################################"

# Install 3ML (current checked out version)
pip install . --upgrade --no-deps

echo "##########################################################"
echo " Installing cthreeML"
echo "##########################################################"

pip install git+https://github.com/giacomov/cthreeML.git

echo "##########################################################"
echo " Setting up HAWC data path and try import HAWC plugin"
echo "##########################################################"

export HAWC_3ML_TEST_DATA_DIR=/hawc_test_data

# Make the matplotlib backend non-interactive (otherwise all tests regarging plotting will fail)
export MPLBACKEND='Agg'

# Try to import the HAWC plugin
python -c "from threeML.plugins.HAWCLike import HAWCLike"
python -c "import os; print(os.environ['HAWC_3ML_TEST_DATA_DIR'])"

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