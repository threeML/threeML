#!/bin/bash
export CPPFLAGS="-I${PREFIX}/include"
export LDFLAGS="-L${PREFIX}/lib"
CC=${PREFIX}/bin/gcc
CXX=${PREFIX}/bin/g++

pip install . -v
