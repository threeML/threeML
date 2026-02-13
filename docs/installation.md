# Installation
3ML brings together multiple instrument and fitting software packages into a
common framework. Thus, installing all the pieces can be a bit of a task for the
user. In order to make this a less painless process, we have packaged most of
the external dependencies into `conda` (see below). However, if you want more
control over your install, 3ML is also available on PyPI via pip. If you have issues
with the installs, first check that you have properly installed all the external
dependencies that *you* plan on using. Are their libraries accessible on you
system's standard paths? If you think that you have everything setup properly
and the install does not work for you, please [submit an
issue](https://github.com/threeML/threeML/issues) and we will do our best to
find a solution.

## In a nutshell:

Run the following commands if you want to install 3ML and astromodels:

* with `conda` without `XSPEC`
```bash
conda install -c threeml -c conda-forge astromodels threeml
```

* with `conda` with `XSPEC`
```bash
conda install -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ -c conda-forge xspec python=3.11
conda install -c threeml -c conda-forge astromodels threeml
```

* with `conda` with `fermitools` and `fermipy`
```bash
conda install -c threeml -c fermi -c conda-forge fermitools astromodels threeml fermipy
```

* with `pip` (please take care of dependencies beforehand!)
```bash
pip install astromodels threeml
```

* in case you want the development versions:
```bash
conda install -c threeml/label/dev -c conda-forge astromodels threeml
```
or
```bash
pip install --upgrade --pre astromodels threeml
```


## Conda installation

[Conda](https://conda.io/docs/) is a platform independent package manager. It
allows to install 3ML (and a lot of other software) without the need to compile
anything, and in a completely separate environment from your system and your
system python.

For detailed conda installation instructions please checkout the 
[documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
Usually `miniconda` is a good start :).

If you are new to conda please stick to these commands and do not change the order of 
the channels for example, those really do matter.


```bash
conda create --name threeML -c conda-forge python=3.11 
```

then activating your environment:

```bash
conda activate threeML
```

```{note}
Recently a third party has uploaded a version of `astromodels` to the
conda-forge channel. This version is not support and  It is important to set your conda channel
priority so that the threeml channel has priority over the conda-forge
channel. Please verify that the `astromodels` installed comes from our threeml
channel. We are working to resolve this hassle and we apologize to our users.
```

Before running the next command, conda will list all packages that will be installed and
from where they will be installed.
Please ensure here that threeml and astromodels both come from something like
```bash
The following packages will be downloaded:
package                    |            build
                        ...
astromodels-2.5.1          |  py311h1831ba8_0         2.1 MB  threeml
                        ...
threeml-2.5.0              |          py311_0        50.0 MB  threeml
                        ...
```
with the important part being the `threeml` at the end of the line. If it says 
`conda-forge` check that the `threeml` channel has a higher priority than the 
`conda-forge` one.

```bash
conda install -c threeml -c conda-forge astromodels threeml
```



If you want to install the dev version add the label dev:
```bash
conda install -c threeml/label/dev -c conda-forge astromodels threeml

```


## pip

If you would like to install 3ML and astromodels on their own and have more
control over which dependencies you would like to use. Please to the following

1. It is highly recommended you work within a python virtual environment to keep
   you base python clean
2. install astromodels

```bash
pip install astromodels
```

3. install 3ML

```bash
pip install threeml
```

If you need to build other dependencies such as pagmo, multinest, XSPEC, etc.,
it is recommended you do this **before** installing astromodels!

If you want to install the dev version add the `--pre` option:
```bash
pip install --upgrade --pre astromodels threeml

```

## XSPEC models
We allow the ability to use the models provided by 
[XSPEC](https://heasarc.gsfc.nasa.gov/docs/software/xspec/index.html) natively in 3ML. 
However, before installing [astromodels](https://astromodels.readthedocs.io) or 3ML, you must 
already have XSPEC installed by one of two methods.

### HEASARC Conda install [recommended]
Since HEASoft 6.35 XSPEC is also distributed as conda package hosted by the HEASARC 
team. This conda packages makes installing XSPEC way easier than building from source
and also for you, wanting to use XSPEC models in threeML.

**Note:** You need to install XSPEC *before* installing `astromodels` and `threeML`:

First let's create the environment and install XSPEC right away. You might rely on 
`xspec-data` as well (caution, that one is large), please check 
[the official instructions](https://heasarc.gsfc.nasa.gov/docs/software/conda.html).

```bash
conda create -n threeML -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ \
  -c conda-forge xspec
```

Perfect, now we have an enviroment called `threeML`. After activating it we can install
`python`
```bash
conda activate threeML
conda install -c conda-forge python=3.11
```

and then you can simply astromodels and threeML using
```bash
conda install -c threeml -c conda-forge astromodels threeml
```
while `-c threeml` before `-c conda-forge` should enforce that the correct 

### HEASARC install
If you have XSPEC installed on your computer via the source code, then before
installing astromodels, make sure to `init` your HEASARC environment. Upon
installing `astromodels`, the process will attempt to find the proper libraries
on your system if it detects XSPEC and compile the proper extensions. If you
have installed your XSPEC in a non-standard directory (e.g. your home
directory), the process my not find all the libraries. You can thus export an
env variable that provides the explicit path to your XSPEC headers:

```bash
export XSPEC_INC_PATH=/path/to/xspec/headers
```

As XSPEC evolves, various models have different interfaces that depend on the
version. To accommodate this, you need to set the version of XSPEC you are using
with another env variable, e.g.:

```bash
export ASTRO_XSPEC_VERSION='12.15.1'
```

## Other dependencies

You need to set up packages such as AERIE (for HAWC), or the Fermi Science
Tools, before running the script, otherwise some of the functionalities will not
work.

* `AERIE` for HAWC: make sure that this works before running the script:

```bash
> liff-PointSourceExpectation --version
INFO [CommandLineConfigurator.cc, ParseCommandLine:137]: 

 liff-PointSourceExpectation
 Aerie version: 2.04.00
 Build type: Debug
    
```
If it doesn't, you need to set up the HAWC environment (refer to the
appropriate documentation)

* `fermitools` and `fermipy` for Fermi/LAT analysis: make sure that this works: 
```bash
> gtirfs ...  P8R2_TRANSIENT100_V6::EDISP0 P8R2_TRANSIENT100_V6::EDISP1 ...
``` 
If it doesn't, you need to install `fermitools` and `fermipy` with conda in the 
same environment:

```bash
conda install -c fermi -c conda-forge fermitools fermipy
```

* `ROOT`: `ROOT` is not required by 3ML, but it provides the Minuit2 minimizer which can 
be used in 3ML. If you have ROOT, make sure that this works before running the script:
```bash
> root-config --version
6.36.06
```

```{note}
There is currently a known incompatibility on some systems between ROOT and 3ML, causing 
the following error while trying to use the ROOT minimizer:

```bash
TypeError: no python-side overrides supported (failed to include Python.h)
```
While this issue is being investigated, we recommend using the `minuit` minimizer directly.

## Install from source (advanced)

Remove any previous installation you might have with:

```bash
> pip uninstall threeML
> pip uninstall astromodels
> pip uninstall cthreeML
```

then:

```bash
> pip install numpy scipy ipython astropy numba cython
> pip install git+https://github.com/threeML/threeml.git  --upgrade
> pip install git+https://github.com/threeML/astromodels.git --upgrade
```

In order to use the HAWC plugin, you will also need to install cthreeML (run
this *after* setting up the HAWC environment):

```bash
> pip install git+https://github.com/threeML/cthreeML.git
```

```{note}
If you do not have permission to install packages in your current python
environment, you can still install the packages by adding the `--user`
option at the end of each `pip` command.
```

### Tips for Mac users
The following paths need to be added to you DYLD_LIBRARY path if you have
FORTRAN installed via these package managers:

* Homebrew: ```DYLD_LIBRARY_PATH=/usr/local/lib/gcc/<version
  number>:$DYLD_LIBRARY_PATH```

* Fink: ```DYLD_LIBRARY_PATH=/sw/lib/gcc<version
  number>/lib:$DYLD_LIBRARY_PATH```

Please inform us if you have problems related to your FORTRAN distribution.

