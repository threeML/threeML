#!/bin/bash

if [ "$(uname)" == "Linux" ]; then
    
    docker run -v `pwd`/../:/3ML --rm -it quay.io/pypa/manylinux1_x86_64 bash -c "source /3ML/conda-dist/build_inside_container.sh"

else
    
    # OS-x
    
    cd recipes
    source activate
    conda build threeml

fi



