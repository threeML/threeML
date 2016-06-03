#!/usr/bin/env python

import os
import sys

from setuptools import setup

# Get the version number
execfile('threeML/version.py')

# Now a global __version__ is available

import imp

# This dynamically loads a module and return it in a variable.
# Will use it for check optional dependencies

def import_module(module_name):

    # Fast path: see if the module has already been imported.

    try:

        return sys.modules[module_name]

    except KeyError:

        pass

    # If any of the following calls raises an exception,
    # there's a problem we can't handle -- let the caller handle it.

    fp, pathname, description = imp.find_module(module_name)

    try:

        return imp.load_module(module_name, fp, pathname, description)

    except:

        raise

    finally:

        # Since we may exit via an exception, close fp explicitly.

        if fp:

            fp.close()

# This list will contain the messages to print just before the end of the setup
# so that the user actually note them, instead of loosing them in the tons of
# messages of the build process

setup(

    name="threeML",

    packages=['threeML',
              'threeML/exceptions',
              'threeML/bayesian',
              'threeML/minimizer',
              'threeML/plugins',
              'threeML/classicMLE',
              'threeML/catalogs',
              'threeML/io',
              'threeML/utils',
              'threeML/parallel',
              'threeML/config'],

    version=__version__,

    description="The Multi-Mission Maximum Likelihood framework",

    long_description="3ML can be used for single or multi-instrument likelihood modeling or Bayesian inference",

    license='BSD-3',

    author='Giacomo Vianello',

    author_email='giacomo.vianello@gmail.com',

    url='https://github.com/giacomov/3ML',

    download_url='https://github.com/giacomov/3ML/archive/%s' % __version__,

    keywords=['Likelihood', 'Multi-mission', '3ML', 'HAWC', 'Fermi', 'HESS', 'joint', 'fit', 'bayesian',
              'multi-wavelength'],

    classifiers=[],

    # Install configuration file in user home and in the package repository

    data_files=[(os.path.join(os.path.expanduser('~'), '.threeML'), ["threeML/config/threeML_config.yml"]),
                ('threeML/config', ["threeML/config/threeML_config.yml"])
                ],

    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'emcee',
        'astropy>=1.0.3',
        'matplotlib',
        'uncertainties',
        'pyyaml',
        'dill',
        'iminuit',
        'astromodels',
        'corner>=1.0.2',
    ])

# Check for optional dependencies

optional_dependencies = {'cthreeML': [False,'needed by HAWC plugin'],
                         'pymultinest': [False, 'needed to use Multinest sampler for Bayesian analysis']}

for dep_name in optional_dependencies:
    
    try:
        
        import_module(dep_name)
    
    except ImportError:
        
        optional_dependencies[dep_name][0] = False
    
    else:
        
        optional_dependencies[dep_name][0] = True

# Now print the final messages

print("\n\n#############")
print("FINAL NOTES:")
print("#############\n\n")

for dep_name in optional_dependencies:
    
    if optional_dependencies[dep_name][0]:
        
        status = 'available'
    
    else:
        
        status = '*NOT* available'
    
    print(" * %s is %s (%s)\n" % (dep_name, status, optional_dependencies[dep_name][1]))
