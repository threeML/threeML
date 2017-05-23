#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    
    LDFLAGS="-lXS -lXSFunctions -lXSModel -lXSUtil" pip install -v .
    
fi

if [ "$(uname)" == "Linux" ]; then

    pip install -v .

fi
