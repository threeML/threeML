![CI](https://github.com/threeML/threeML/workflows/CI/badge.svg?branch=master&event=push)
[![Conda Build and Publish](https://github.com/threeML/threeML/actions/workflows/conda_build.yml/badge.svg)](https://github.com/threeML/threeML/actions/workflows/conda_build.yml)
![Test Against XSPEC](https://github.com/threeML/threeML/workflows/Test%20Against%20XSPEC/badge.svg)
[![codecov](https://codecov.io/gh/threeML/threeML/branch/master/graph/badge.svg)](https://codecov.io/gh/threeML/threeML)
[![Documentation Status](https://readthedocs.org/projects/threeml/badge/?version=latest)](http://threeml.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5646954.svg)](https://doi.org/10.5281/zenodo.5646954)

![GitHub pull requests](https://img.shields.io/github/issues-pr/threeML/threeML)
![GitHub issues](https://img.shields.io/github/issues/threeML/threeML)

## PyPi

[![PyPI version fury.io](https://badge.fury.io/py/threeML.svg)](https://pypi.python.org/pypi/threeML/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/threeml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/threeml)
[![Install using pip](https://github.com/threeML/threeML/actions/workflows/pip_install.yml/badge.svg)](https://github.com/threeML/threeML/actions/workflows/pip_install.yml)
## Conda

![Conda](https://img.shields.io/conda/pn/threeml/threeml)
![Conda](https://img.shields.io/conda/dn/threeml/threeml)

<div  >
<img src="https://raw.githubusercontent.com/threeML/threeML/master/logo/logo_sq.png" alt="drawing" width="300" align="right"/>
<header >
  <h1>
   <p > The Multi-Mission Maximum Likelihood framework (3ML)</p>
  </h1>
</header>

A framework for multi-wavelength/multi-messenger analysis for astronomy/astrophysics.

<br/>
</div>


Astrophysical sources are observed by different instruments at different
wavelengths with an unprecedented quality. Putting all these data together to
form a coherent view, however, is a very difficult task. Indeed, each instrument
and data type has its own ad-hoc software and handling procedure, which present
steep learning curves and do not talk to each other.

The Multi-Mission Maximum Likelihood framework (3ML) provides a common
high-level interface and model definition, which allows for an easy, coherent
and intuitive modeling of sources using all the available data, no matter their
origin. At the same time, thanks to its architecture based on plug-ins, 3ML uses
under the hood the official software of each instrument, the only one certified
and maintained by the collaboration which built the instrument itself. This
guarantees that 3ML is always using the best possible methodology to deal with
the data of each instrument.

<img src="https://raw.githubusercontent.com/threeML/threeML/master/docs/media/3ml_flowchart.png" alt="drawing" width="800" align="right"/>


Though **Maximum Likelihood** is in the name for historical reasons, 3ML is an
interface to several **Bayesian** inference algorithms such as MCMC and nested
sampling as well as likelihood optimization algorithms. Each approach to
analysis can be seamlessly switched between allowing users to try different
approaches quickly and without having to rewrite their model or data interfaces.

Like your [XPSEC](https://heasarc.gsfc.nasa.gov/xanadu/xspec/) models? You can
use them in 3ML as well as our growing selection of 1-,2- and 3-D models from
our fast and customizable modeling language
[astromodels](http://astromodels.readthedocs.org/en/latest/).


## Installation

Installing with pip or conda is easy. However, you want to include models from
[XSPEC](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html), the
process can get tougher and we recommend the more detailed instructions:


```bash
pip install astromodels threeml
```

```bash
conda  install astromodels threeml -c threeml conda-forge 
```
Please refer to the [Installation instructions](https://threeml.readthedocs.io/en/stable/notebooks/installation.html) for more details and trouble-shooting.

## Press
* [Software in development at Stanford advances the modeling of astronomical observations](https://news.stanford.edu/2017/12/07/software-advances-modeling-astronomical-observations/)

## Who is using 3ML?
Here is a highlight list of teams and their publications using 3ML.

* [Fermi-LAT](https://fermi.gsfc.nasa.gov) and [Fermi-GBM](https://grb.mpe.mpg.de)
  * [GeV–TeV Counterparts of SS 433/W50 from Fermi-LAT and HAWC Observations](https://iopscience.iop.org/article/10.3847/2041-8213/ab62b8)
  * [The Bright and the Slow](https://iopscience.iop.org/article/10.3847/1538-4357/aad6ea)
* [HAWC](https://www.hawc-observatory.org)
  * [Extended gamma-ray sources around pulsars constrain the origin of the positron flux at Earth](https://science.sciencemag.org/content/358/6365/911)
  * [Evidence of 200 TeV photons from HAWC J1825-134](https://arxiv.org/abs/2012.15275)
* [POLAR](https://www.astro.unige.ch/polar-2/?fbclid=IwAR0IxMxHtiXZyqc0A_kT1xKe9ASAk_VmfJpCEBr0HOhDG5eOHY7AE5TWHv8)
  * [The POLAR gamma-ray burst polarization catalog](https://ui.adsabs.harvard.edu/link_gateway/2020A&A...644A.124K/doi:10.1051/0004-6361/202037915)

A full list of publications using 3ML is [here](https://ui.adsabs.harvard.edu/abs/2015arXiv150708343V/citations).

## Citing 
If you find this package useful in you analysis, or the code in your own custom data tools, please cite:

[Vianello et al. (2015)](https://arxiv.org/abs/1507.08343)



### Acknowledgements 
3ML makes use of the Spanish Virtual Observatory's Filter Profile service (http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=NIRT).

If you use these profiles in your research, please consider citing them by using the following suggested sentence in your paper:

"This research has made use of the SVO Filter Profile Service (http://svo2.cab.inta-csic.es/theory/fps/) supported from the Spanish MINECO through grant AyA2014-55216"

and citing the following publications:

The SVO Filter Profile Service. Rodrigo, C., Solano, E., Bayo, A. http://ivoa.net/documents/Notes/SVOFPS/index.html
The Filter Profile Service Access Protocol. Rodrigo, C., Solano, E. http://ivoa.net/documents/Notes/SVOFPSDAL/index.html


<img src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png"  width="100"> ThreeML is supported by National Science Foundation (NSF) <img src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png"  width="100">

