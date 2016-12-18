[![Build Status](https://travis-ci.org/giacomov/3ML.svg?branch=master)](https://travis-ci.org/giacomov/3ML)
[![Coverage Status](https://coveralls.io/repos/github/giacomov/3ML/badge.svg?branch=master)](https://coveralls.io/github/giacomov/3ML?branch=master)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/giacomov/3ml)
[![Code Climate](https://codeclimate.com/github/giacomov/3ML/badges/gpa.svg)](https://codeclimate.com/github/giacomov/3ML)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength analysis for astronomy/astrophysics. See the [3ML website](https://threeml.stanford.edu) 


# License

BSD-3

# Try-before-install
You can try 3ML without installing or downloading anything on your computer. 

Thanks to [mybinder.org](mybinder.org), we provide a notebook environment that you can access from your 
web browser. With it you can run some of the basic examples and experiment on your own 
using sample datasets available in the ```examples``` directory.

Simply enter the ```examples``` directory then click on the basic_test.ipynb notebook, 
or create your own.

If you are new or need a refresher on how to use the jupyter notebook, see 
[[here](https://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Notebook%20Basics.ipynb)].

> NOTE: the test environment does not provide all functionalities. For example, 
MULTINEST, Xspec models and parallel computation are not supported, and only the default
minimizer ([iminuit](https://github.com/iminuit/iminuit)) is available.

To start, click here: 

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/giacomov/3ml)

# Installation

Remove any previous installation you might have with:

```bash
> pip uninstall threeML
> pip uninstall astromodels
> pip uninstall cthreeML
```

then:

```bash
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

