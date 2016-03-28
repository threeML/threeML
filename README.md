# The Multi-Mission Maximum Likelihood framework (3ML)

A framework for multi-wavelength analysis for astronomy/astrophysics. See the [3ML website](https://threeml.stanford.edu) 


# License

BSD-3

# Installation

Clone the repository with:

```
> git clone https://github.com/giacomov/3ML.git
```

## Build 3ML without support for C/C++ plugins (default, for most users)

At the moment only the HAWC plugin needs the C/C++ wrapper interface. If you are not interested in using HAWC data, it is strongly adviced that you build 3ML without support for C/C++ plugins, by simply doing:

```
> python setup.py install
```

## For HAWC: build 3ML with support for C/C++ plugins

If you are interested in HAWC analysis, you need C/C++ support and of course you need the HAWC AERIE software installed and set up first. Then, in order to build 3ML with C/C++ support you need the Boost::python library, which comes with AERIE. Set the BOOSTROOT environment variables to the directory containing boost (it is under the Ape externals, and it contains a "lib" and an "include" directory):

```
> export BOOSTROOT=[path to the boost directory]
```

Then build 3ML with boost:

```
> python setup.py install --with-boost
```
At the end you should see a message like this:

```
#############
FINAL NOTES:
#############
Built the boost.python extension.
Used boost.python from the env. variable BOOSTROOT
     Include dir: [...]
     Library dir: [...]

```



