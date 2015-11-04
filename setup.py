#!/usr/bin/env python
 
#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from distutils.command.install_headers import install_headers
import os, sys

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

#This finds root configuration
from subprocess import Popen, PIPE
def root_config(*opts):
    args = ['root-config'] + ['--'+o for o in opts]
    
    try:
        
	config = Popen(args, stdout=PIPE).communicate()[0].strip().split(' ')
    
    except:
        
	print("You need ROOT installed")
	
	sys.exit(-1)
    
    else:
        
	return config

###########################################################
#Get ROOT informations

incdirs = []
libdirs = []
libs = ["Minuit2"]

version, incdir, libdir = root_config('version', 'incdir', 'libdir')
incdirs.append(incdir)
libdirs.append(libdir)
libs += ["Core","Cint","RIO","Net","Hist","Graf","Rint","Matrix","MathCore"]
print("Linking against Minuit2 library from ROOT %s" % version)


	
setup(
    
    name="threeML",
    
    packages = ['threeML',
                'threeML/exceptions',
                'threeML/bayesian',
                'threeML/minimizer',
                'threeML/models',
                'threeML/models/fluxModels',
                'threeML/models/spatialModels',
                'threeML/plugins',
                'threeML/classicMLE',
                'threeML/catalogs',
                'threeML/io',
                'threeML/utils',
                'threeML/parallel',
                'threeML/config'],
    
    version = 'v0.1.0',
    
    description = "The Multi-Mission Maximum Likelihood framework",
    
    author = 'Giacomo Vianello',
    
    author_email = 'giacomo.vianello@gmail.com',
    
    url = 'https://github.com/giacomov/3ML',
    
    download_url = 'https://github.com/giacomov/3ML/archive/v0.1.0',
    
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
        
        #Extension("threeML.libModelInterface", 
                  
        #          ["threeML/models/pyToCppModelInterface.cxx",
        #           "threeML/models/FixedPointSource.cxx"],
        
        
        
        #          libraries = ["boost_python"],
        
        #          include_dirs=include_dirs,
        
        #          library_dirs=library_dirs),
	
	Extension("threeML.minuit2",
	          
		  ["threeML/minuit2/minuit2.cpp"],
		  
		  library_dirs=libdirs,
                  libraries=libs,
                  include_dirs=incdirs
		  
		  )
		  
    ],
    
    
    headers=["threeML/models/ModelInterface.h",
             "threeML/models/pyToCppModelInterface.h",
             "threeML/models/FakePlugin.h",
             "threeML/models/FixedPointSource.h"],
    
    #Install configuration file in user home and in the package repository
    
    data_files = [( os.path.join( os.path.expanduser( '~' ), '.threeML' ) ,["threeML/config/threeML_config.yml"]),
                  ( os.path.join( sys.prefix, 'threeML/config' ), ["threeML/config/threeML_config.yml"])
                  ],
    
    install_requires=[
          'numpy >= 1.6',
          'scipy',
          'numexpr',
          'emcee',
          'astropy >= 1.0.0',
          'matplotlib',
          'ipython >= 2.0.0, < 3.9.9',
          'uncertainties',
          'pyyaml',
	  'dill',
	  'parse'
      ])

