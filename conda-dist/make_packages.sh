#!/bin/bash

# Remember to set these:
# CONDA_ENV='/home/giacomov/miniconda2'
# WHEEL2CONDA='/home/giacomov/.local/bin/wheel2conda'

if [ "$(uname)" == "Linux" ]; then
    
    unset HEADAS
    unset LD_PRELOAD
    
    # activate conda
    source ${CONDA_ENV}/bin/activate
    
    echo "Using python:"
    which python
    
    echo "Make wheels"
    
    python make_wheels.py
    
    echo "Run wheel2conda"
    
    for wheel in *.whl
    do
        
        echo Processing $wheel
        
        $WHEEL2CONDA $wheel
        
        # Remove all packages not for python 2.7
        rm -rf linux-64/*py34*
        rm -rf linux-64/*py35*
        rm -rf osx-64/*py34*
        rm -rf osx-64/*py35*
        
    done
    
    rm -rf win-32
    rm -rf win-64
    rm -rf linux-32
    rm -rf *.whl

    # Remove threeML (it will be uploaded later in noarh)
    rm -rf linux-64/three*
    rm -rf osx-64/three*

    echo "Add the wheels to the channel"
    anaconda upload linux-64/* --force
    anaconda upload osx-64/* --force

fi

echo "Make custom recipes"
conda config --set anaconda_upload yes
conda build recipes/* -c giacomov


