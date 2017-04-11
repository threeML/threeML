[![Build Status](https://travis-ci.org/giacomov/3ML.svg?branch=master)](https://travis-ci.org/giacomov/3ML)
[![codecov](https://codecov.io/gh/giacomov/3ML/branch/master/graph/badge.svg)](https://codecov.io/gh/giacomov/3ML)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/giacomov/3ml)
[![Code Climate](https://codeclimate.com/github/giacomov/3ML/badges/gpa.svg)](https://codeclimate.com/github/giacomov/3ML)
[![Documentation Status](https://readthedocs.org/projects/threeml/badge/?version=latest)](http://threeml.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength/multi-messenger analysis for astronomy/astrophysics.

# Try-before-install
You can try 3ML without installing or downloading anything on your computer. 

Thanks to [mybinder.org](mybinder.org), we provide a notebook environment that you can access from your 
web browser. With it you can run some of the basic examples and experiment on your own 
using sample datasets available in the ```examples``` directory.

Once in binder, simply navigate to the ```examples``` directory then click on the basic_test.ipynb notebook, 
or create your own.

If you are new or need a refresher on how to use the jupyter notebook, see 
[[here](https://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Notebook%20Basics.ipynb)].

> NOTE: the test environment does not provide all functionalities. For example, 
MULTINEST, Xspec models and parallel computation are not supported, and only the default
minimizer ([iminuit](https://github.com/iminuit/iminuit)) is available.

To start, click here: 

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/giacomov/3ml)

# Installation

## Automatic script (suggested)

We provide a script which automates the installation process. It creates a python 
virtual environment which contains all the packages required to run 3ML. 

> A virtual
environment is a directory which contains the files needed for a set of packages 
(in this case 3ML and its 
dependencies) to work. By using it, we avoid touching any pre-existing python 
environment you have. If something goes wrong, we can just remove the directory 
containing the python virtual environment and restart, without ever touching the 
system python. A virtual environment must be activated before use, as demonstrated
below. After activation, running ```pip install [package]``` will install the package
within the virtual environment.

#### Pre-requisites for the script

You likely have these already. However, spend a few minutes making sure to avoid 
problems down the line. You need:
 
1. a working python 2.7 environment (python3 is not 
supported yet because many of the mission packages, like the Fermi Science Tools, need 
python2.7). This is installed by default on most modern operating systems.

2. The python package ```virtualenv```. Verify that you can run this command:
    ```bash
    > virtualenv --version
    ```
    If the command does not exist, you can install virtualenv by running 
    ```sudo pip install virtualenv```. If you do 
    not have administrative priviliges, you can still install it with 
    ```pip install --user virtualenv```, but then you need to add ```~/.local/bin``` to your
    ```PATH``` environment variable.

3. The ```git``` versioning system. You can obtain this from your operating system 
provider.

Before continuing, make sure that these 3 commands work and provide an output similar
to what presented here (versions might of course be a little different):
```bash
> python --version
Python 2.7.12
> virtualenv --version
15.1.0
> git --version
git version 2.7.4
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

* XSPEC models: to use Xspec models within 3ML, you need to have a working Xspec 
installation. Make sure that this works:
    ```bash
    > echo exit | xspec
    
    		XSPEC version: 12.9.0n
    	Build Date/Time: Sat Sep 17 00:43:48 2016
    
    XSPEC12>exit
     XSPEC: quit
    ```

* ROOT: ROOT is not required by 3ML, but it provides the Minuit2 minimizer which can 
be used in 3ML. If you have ROOT, make sure that this works before running the script:
    ```bash
    > root-config --version
    5.34/36
    ```

* MULTINEST: you need to download and compile [multinest](https://github.com/JohannesBuchner/MultiNest) , 
then *after running the 3ML installation script*, you need to run *within the virtual 
environment* ```pip install pymultinest```


### Download and run the script

Download the installation script with:
```bash
> python -c "import urllib ; urllib.urlretrieve('https://raw.githubusercontent.com/giacomov/3ML/master/install_3ML.py','install_3ML.py')"
```
You can also access it directly from [this link](https://raw.githubusercontent.com/giacomov/3ML/master/install_3ML.py).

Run the script:
```bash
> python install_3ML.py
```
and follow the instructions.

The script will install astromodels, 3ML and (if the setup script will find 
boost::python) cthreeML. Note that if you have AERIE installed and set up, boost::python
is included and cthreeML will be installed by default.

### Using 3ML
Before you can run 3ML, you need to activate the virtual environment. Assuming you used
the default name, this is achieved with:
```bash
> source ~/3ML_env/bin/activate
```
You should see that the prompt changes to something like:
```bash
(3ML_env) >
```

For performance optimization, you can also set these env. variables *after activating
the virtual environment*:
```bash
(3ML_env) > export OMP_NUM_THREADS=1
(3ML_env) > export MKL_NUM_THREADS=1
(3ML_env) > export NUMEXPR_NUM_THREADS=1
```

It is probably a good idea to place these instructions in a init script that you can
source, or in your .bashrc file.

This is a minimal example of such file, which we can call activate_3ML.sh:
```bash
source ~/3ML_env/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```



You can now use 3ML.

> NOTE: of course, before activating the virtual environment you need to repeat all 
the steps needed to configure the software you want to use in 3ML (AERIE, the
Fermi Science Tools, Xspec...).

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
> pip install git+https://github.com/giacomov/astromodels.git
```

In order to use the HAWC plugin, you will also need to install cthreeML (run this *after* setting up the HAWC environment):

```bash
> pip install git+https://github.com/giacomov/cthreeML.git
```

* NOTE: If you do not have permission to install packages in your current python 
environment, you can still install the packages by adding the ```--user``` option at the
end of each ```pip``` command.

#### Tips for Mac users
The following paths need to be added to you DYLD_LIBRARY path if you have FORTRAN installed via these package managers:

* Homebrew: DYLD_LIBRARY_PATH=/usr/local/lib/gcc/<version number>:$DYLD_LIBRARY_PATH
* Fink: DYLD_LIBRARY_PATH=/sw/lib/gcc<version number>/lib:$DYLD_LIBRARY_PATH

Please inform us if you have problems related to your FORTRAN distribution.

### Acknowledgements 
3ML makes use of the Spanish Virtual Observatory's Filter Profile servce (http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=NIRT).

If you use these profiles in your research, please consider citing them:

This research has made use of the SVO Filter Profile Service (http://svo2.cab.inta-csic.es/theory/fps/) supported from the Spanish MINECO through grant AyA2014-55216
and we would appreciate if you could include the following references in your publication:

The SVO Filter Profile Service. Rodrigo, C., Solano, E., Bayo, A. http://ivoa.net/documents/Notes/SVOFPS/index.html
The Filter Profile Service Access Protocol. Rodrigo, C., Solano, E. http://ivoa.net/documents/Notes/SVOFPSDAL/index.html


### Citing 3ML





