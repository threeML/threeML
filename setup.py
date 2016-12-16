#!/usr/bin/env python

import os
import sys

from setuptools import setup

from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    test_package_name = 'threeML'

    def finalize_options(self):
        TestCommand.finalize_options(self)
        _test_args = [
            '--verbose',
            '--ignore=build',
            '--cov={0}'.format(self.test_package_name),
            '--cov-report=term',
            # '--pep8',
        ]
        extra_args = os.environ.get('PYTEST_EXTRA_ARGS')
        if extra_args is not None:
            _test_args.extend(extra_args.split())
        self.test_args = _test_args
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        self.handle_exit()

        sys.exit(errno)

    @staticmethod
    def handle_exit():
        import atexit

        atexit._run_exitfuncs()



# Get the version number
execfile('threeML/version.py')

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
              'threeML/plugins/OGIP',
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

    # data_files=[(os.path.join(os.path.expanduser('~'), '.threeML'), ["threeML/config/threeML_config.yml"]),
    #             ('threeML/config', ["threeML/config/threeML_config.yml"])
    #             ],

    package_data={'threeML': ['config/*.yml'],},
    include_package_data=True,

    install_requires=[
        'numpy >= 1.6',
        'scipy',
        'emcee',
        'astropy>=1.0.3',
        'matplotlib',
        'uncertainties',
        'pyyaml',
        'dill',
        'iminuit>=1.2',
        'astromodels>=0.2.1',
        'corner>=1.0.2',
        'pandas',
        'sympy'
    ],

        tests_require=['pytest', 'pytest-cov'],
        cmdclass={'test': PyTest})

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
