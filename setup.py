#!/usr/bin/env python

import os
import sys

import glob

from setuptools import setup

# Get the version number
with open("threeML/version.py") as f:
    version_code = compile(f.read(), "threeML/version.py", 'exec')
    exec(version_code)

# Now a global __version__ is available

# This dynamically loads a module and return it in a variable.
# Will use it for check optional dependencies

def is_module_available(module_name):

    # Fast path: see if the module has already been imported.

    try:

        exec('import %s' % module_name)

    except ImportError:

        return False

    else:

        return True


# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join('..', path, filename))

    return paths

extra_files = find_data_files('threeML/data')

# This list will contain the messages to print just before the end of the setup
# so that the user actually note them, instead of loosing them in the tons of
# messages of the build process

setup(

    name="threeML",

    packages=['threeML',
              'threeML/exceptions',
              'threeML/bayesian',
              'threeML/minimizer',
              'threeML/utils',
              'threeML/utils/OGIP',
              'threeML/utils/spectrum',
              'threeML/utils/photometry',
              'threeML/utils/time_series',
              'threeML/utils/data_builders',
              'threeML/utils/data_builders/fermi',
              'threeML/utils/data_download',
              'threeML/utils/data_download/Fermi_LAT',
              'threeML/utils/data_download/Fermi_GBM',
              'threeML/utils/fitted_objects',
              'threeML/utils/statistics',
              'threeML/plugins',
              'threeML/classicMLE',
              'threeML/catalogs',
              'threeML/io',
              'threeML/io/plotting',
              'threeML/io/cern_root_utils',
              'threeML/parallel',
              'threeML/config',
              'threeML/test',
              'threeML/plugins/experimental'
              ],

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

    # data_files=[(os.path.join(os.path.expanduser('~'), '.threeML'), ["threeML/config/threeML_config.yml"]),
    #             ('threeML/config', ["threeML/config/threeML_config.yml"])
    #             ],

    # NOTE: we use '' as package name because the extra_files already contain the full path from here

        package_data={'': extra_files, },
    include_package_data=True,

    install_requires=[
        'numpy >= 1.6',
        'scipy >=0.18',
        'emcee',
        'astropy>=1.3.3',
        'astroquery',
        'matplotlib',
        'uncertainties',
        'pyyaml',
        'dill',
        'iminuit>=1.2',
        'astromodels>=0.4.0',
        'corner>=1.0.2',
        'pandas',
        'requests',
        'speclite',
        'ipython<=5.9'
    ],

    extras_require={
            'tests': [
                'pytest',],
            'docs': [
                'sphinx >= 1.4',
                'sphinx_rtd_theme',
                'nbsphinx']}

    ) # End of setup()

# Check for optional dependencies

optional_dependencies = {'cthreeML': [False,'needed by HAWC plugin'],
                         'pymultinest': [False, 'provides the Multinest sampler for Bayesian analysis'],
                         'pyOpt': [False, 'provides more optimizers'],
                         'ROOT': [False, 'provides the ROOT optimizer'],
                         'ipywidgets': [False, 'provides widget for jypyter (like the HTML progress bar)']}

for dep_name in optional_dependencies:

    optional_dependencies[dep_name][0] = is_module_available(dep_name)

# Now print the final messages

print("\n\n##################")
print("OPTIONAL FEATURES:")
print("##################\n\n")

for dep_name in optional_dependencies:
    
    if optional_dependencies[dep_name][0]:
        
        status = 'available'
    
    else:
        
        status = '*NOT* available'
    
    print(" * %s is %s (%s)\n" % (dep_name, status, optional_dependencies[dep_name][1]))
