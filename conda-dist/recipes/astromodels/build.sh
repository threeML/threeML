#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    
    if [ -z ${HEASOFT+x} ]; then

        pip install -v .
    
    else
 
        LDFLAGS="-lXS -lXSFunctions -lXSModel -lXSUtil" pip install -v .
    
    fi
    
fi

if [ "$(uname)" == "Linux" ]; then

    pip install -v .

fi
