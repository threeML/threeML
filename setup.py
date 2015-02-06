#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
 
include_dirs = [ '/home/giacomov/software/boost/boost_1_57_0/']
 
library_dirs = [ '/home/giacomov/software/boost/boost_1_57_0/bin.v2/libs/python/build/gcc-4.7/debug/' ]

 
setup(name="threeML",
    packages = ['threeML'],
    version = '0.1',
    description = "The Multi-Mission Maximum Likelihood framework",
    author = 'Giacomo Vianello',
    author_email = 'giacomov@stanford.edu',
    url = 'https://github.com/giacomov/3ML',
    download_url = 'https://github.com/giacomov/3ML/tarball/0.1',
    keywords = [],
    classifiers = [],
    ext_modules=[
        Extension("ModelInterface", ["ModelInterface.cxx","ModelInterface_boost.cxx"],
        libraries = ["boost_python"],
        include_dirs=include_dirs,
        library_dirs=library_dirs)
    ])

