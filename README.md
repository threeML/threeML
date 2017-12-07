[![Build Status](https://travis-ci.org/giacomov/3ML.svg?branch=master)](https://travis-ci.org/giacomov/3ML)
[![codecov](https://codecov.io/gh/giacomov/3ML/branch/master/graph/badge.svg)](https://codecov.io/gh/giacomov/3ML)
[![Code Climate](https://codeclimate.com/github/giacomov/3ML/badges/gpa.svg)](https://codeclimate.com/github/giacomov/3ML)
[![Documentation Status](https://readthedocs.org/projects/threeml/badge/?version=latest)](http://threeml.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength/multi-messenger analysis for astronomy/astrophysics.

# Acknowledgements 
3ML makes use of the Spanish Virtual Observatory's Filter Profile servce (http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=NIRT).

If you use these profiles in your research, please consider citing them:

This research has made use of the SVO Filter Profile Service (http://svo2.cab.inta-csic.es/theory/fps/) supported from the Spanish MINECO through grant AyA2014-55216
and we would appreciate if you could include the following references in your publication:

The SVO Filter Profile Service. Rodrigo, C., Solano, E., Bayo, A. http://ivoa.net/documents/Notes/SVOFPS/index.html
The Filter Profile Service Access Protocol. Rodrigo, C., Solano, E. http://ivoa.net/documents/Notes/SVOFPSDAL/index.html

# Installation

## Conda installation (suggested)

[Conda](https://conda.io/docs/) is a platform independent package manager. It allows to install 3ML (and a lot of other software) without the need
to compile anything, and in a completely separate environment from your system and your system python.

### If you don't know Conda

If you are not familiar with conda, install 3ML with the automatic script which will take care of everything:

1. Download the script from [here](https://github.com/giacomov/3ML/blob/master/install_3ML.sh)
2. Run the script with `bash install_3ML.sh`
3. The script will install 3ML and then create a `threeML_init.sh` script and a `threeML_init.csh` script. Source the former if you are using Bash
(`source threeML_init.sh`) and the second one if you are using Csh/Tcsh (`source threeML_init.csh`).

### If you already know Conda 

If you are familiar with Conda and you already have it installed, you can install threeML by creating an environment with:

```bash
conda create --name threeML -c conda-forge python=2.7 numpy scipy matplotlib
```

then activating your environment and installing 3ML as:

```bash
source activate threeML
conda install -c conda-forge -c threeml threeml
```

### Other dependencies

You need to set up packages such as AERIE (for HAWC), or the Fermi Science Tools, 
before running the script, otherwise some of the functionalities will not work.

* AERIE for HAWC: make sure that this works before running the script:

    ```bash
    > liff-PointSourceExpectation --version
    INFO [CommandLineConfigurator.cc, ParseCommandLine:137]: 
    
     liff-PointSourceExpectation
     Aerie version: 2.04.00
     Build type: Debug
    
    ```
    If it doesn't, you need to set up the HAWC environment (refer to the appropriate 
    documentation)

* Fermi Science Tools for Fermi/LAT analysis: make sure that this works:
    ```bash
    > gtirfs
    ...
    P8R2_TRANSIENT100_V6::EDISP0
    P8R2_TRANSIENT100_V6::EDISP1
    ...
    ```
    If it doesn't, you need to configure and set up the Fermi Science Tools.

* ROOT: ROOT is not required by 3ML, but it provides the Minuit2 minimizer which can 
be used in 3ML. If you have ROOT, make sure that this works before running the script:
    ```bash
    > root-config --version
    5.34/36
    ```

## Install using pip (advanced)

Since this method alters the python environment you have on your system, 
we suggest you use this method only if you understand the implications.

Remove any previous installation you might have with:

```bash
> pip uninstall threeML
> pip uninstall astromodels
> pip uninstall cthreeML
```

then:

```bash
> pip install numpy scipy ipython
> pip install git+https://github.com/giacomov/3ML.git 
> pip install git+https://github.com/giacomov/astromodels.git --upgrade
```

In order to use the HAWC plugin, you will also need to install cthreeML (run this *after* setting up the HAWC environment):

```bash
> pip install git+https://github.com/giacomov/cthreeML.git
```

* NOTE: If you do not have permission to install packages in your current python 
environment, you can still install the packages by adding the ```--user``` option at the
end of each ```pip``` command.

### Tips for Mac users
The following paths need to be added to you DYLD_LIBRARY path if you have FORTRAN installed via these package managers:

* Homebrew: DYLD_LIBRARY_PATH=/usr/local/lib/gcc/<version number>:$DYLD_LIBRARY_PATH
* Fink: DYLD_LIBRARY_PATH=/sw/lib/gcc<version number>/lib:$DYLD_LIBRARY_PATH

Please inform us if you have problems related to your FORTRAN distribution.


### Citing 3ML





