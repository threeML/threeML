[![Build Status](https://travis-ci.org/giacomov/3ML.svg?branch=master)](https://travis-ci.org/giacomov/3ML)
[![Coverage Status](https://coveralls.io/repos/github/giacomov/3ML/badge.svg?branch=master)](https://coveralls.io/github/giacomov/3ML?branch=master)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/giacomov/3ml)
# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength analysis for astronomy/astrophysics. See the [3ML website](https://threeml.stanford.edu) 


# License

BSD-3

# Try-before-install
You can try 3ML without installing or downloading anything on your computer. 

Thanks to mybinder.org, we provide a test environment that you can access from your web
browser. In it you can navigate to the examples directory and run some of the basic
examples. 

Simply enter the "examples" directory then click
on the basic_test.ipynb notebook.

If you are new or need a refresher on how to use the jupyter notebook, see 
[[here](https://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Notebook%20Basics.ipynb)].

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
> pip install git+https://github.com/giacomov/astromodels.git
```

* NOTE: If you do not have permission to install packages in your current python 
environment, you can still install the packages by adding the ```--user``` option at the
end of each ```pip``` command.

