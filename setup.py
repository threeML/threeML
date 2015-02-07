#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_headers import install_headers
import os

boost_root = os.environ.get("BOOSTROOT")

if(boost_root):
  #The user want to override pre-defined location of boost
  
  print("\n\n **** Using boost.python from the env. variable $BOOSTROOT (%s)" %(boost_root))
  
  include_dirs = [ os.path.join(boost_root,'include')]
  library_dirs = [ os.path.join(boost_root,'lib') ]
  
  print("     Include dir: %s" %(include_dirs))
  print("     Library dir: %s" %(library_dirs))

else:

  include_dirs = []
  library_dirs = []

pass
 
setup(
    
    name="threeML",
    
    packages = ['threeML',
                'threeML/bayesian',
                'threeML/minimizer',
                'threeML/models',
                'threeML/plugins'],
    
    version = 'v0.0.5',
    
    description = "The Multi-Mission Maximum Likelihood framework",
    
    author = 'Giacomo Vianello',
    
    author_email = 'giacomo.vianello@gmail.com',
    
    url = 'https://github.com/giacomov/3ML',
    
    download_url = 'https://github.com/giacomov/3ML/archive/v0.0.5',
    
    keywords = ['Likelihood','Multi-mission','3ML','HAWC','Fermi','joint','fit'],
    
    classifiers = [],
    
    ext_modules=[
        
        Extension("threeML.pyModelInterface", 
                  
                  ["threeML/models/pyToCppModelInterface.cxx",
                   "threeML/models/FixedPointSource.cxx",
                   "threeML/models/ModelInterface_boost.cxx"],
        
        
        
        libraries = ["boost_python"],
        
        include_dirs=include_dirs,
        
        library_dirs=library_dirs),
        
        Extension("threeML.libModelInterface", 
                  
                  ["threeML/models/pyToCppModelInterface.cxx",
                   "threeML/models/FixedPointSource.cxx"],
        
        
        
        libraries = ["boost_python"],
        
        include_dirs=include_dirs,
        
        library_dirs=library_dirs),
    ],
    
    
    headers=["threeML/models/ModelInterface.h",
             "threeML/models/pyToCppModelInterface.h",
             "threeML/models/FakePlugin.h",
             "threeML/models/FixedPointSource.h"],
    
    install_requires=[
          'numpy',
          'scipy',
          'numexpr',
          'numdifftools',
          'emcee',
          'pyfits',
          'matplotlib',
          'ipython>=2.0.0, <3.0.0'          
      ])

