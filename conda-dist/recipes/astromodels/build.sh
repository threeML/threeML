#!/bin/bash

CC=${PREFIX}/bin/gcc
CXX=${PREFIX}/bin/g++

if [ "$(uname)" == "Darwin" ]; then
    
    if [ -z ${HEASOFT+x} ]; then

        pip install -v .
    
    else
 
        LDFLAGS="-lXS -lXSFunctions -lXSModel -lXSUtil" pip install -v .
    
    fi
    
fi

if [ "$(uname)" == "Linux" ]; then

    LDFLAGS="-L${PREFIX} -lgfortran" pip install -v .

fi
