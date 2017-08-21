#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_{C,CXX}_FLAGS="-arch x86_64" -DCMAKE_Fortran_FLAGS="-m64" ..
    make
    make install
    
fi

if [ "$(uname)" == "Linux" ]; then

    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} -DLAPACK_LIBRARIES=${PREFIX}/lib/liblapack.so -DBLAS_LIBRARIES=${PREFIX}/lib/libcblas.so ..
    make
    make install

fi
