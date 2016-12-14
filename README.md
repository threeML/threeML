[![Build Status](https://travis-ci.org/giacomov/3ML.svg?branch=master)](https://travis-ci.org/giacomov/3ML)

# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength analysis for astronomy/astrophysics. See the [3ML website](https://threeml.stanford.edu) 


# License

BSD-3

# Installation

Remove any previous installation you might have with:

```
> pip uninstall threeML
> pip uninstall astromodels
> pip uninstall cthreeML
```

then:

```
> pip install threeML --user
```

In order to use the HAWC plugin, you will also need to install cthreeML (run this *after* setting up the HAWC environment):

```
> pip install cthreeML --user
```

* NOTE: if you want to install the packages system-wide, remove the '--user' option

