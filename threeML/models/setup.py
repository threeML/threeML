#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
 
include_dirs = [ '/home/giacomov/software/boost/boost_1_57_0/']
 
library_dirs = [ '/home/giacomov/software/boost/boost_1_57_0/bin.v2/libs/python/build/gcc-4.7/debug/' ]

 
setup(name="PackageName",
    ext_modules=[
        Extension("ModelInterface", ["ModelInterface.cxx","ModelInterface_boost.cxx"],
        libraries = ["boost_python"],
        include_dirs=include_dirs,
        library_dirs=library_dirs)
    ])
