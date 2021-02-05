---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Generating Synthetic Data
In data analysis, it is important that we have the ability to test our assumptions. One powerful tool to enable these tests is simulation. In 3ML, we have several ways to generate synthetic data sets both from models and from fits.


## Synthetic data from spectra


### Genertating data from models

Most of the current plugins support the ability to generate synthetic data directly from a model. This can be very useful to assertain the detectability of a source/component/line or simply to see how models look once they are transformed into data. Below we will demonstrate how different plugins transform a model into synthetic data.


#### XYLike

In many of the examples, the basic XYLike plugin has been used to generate synthetic data. Here, we will revisit the plugin for completeness.


```python nbsphinx="hidden"
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
```


```python
import matplotlib.pyplot as plt
import numpy as np

from threeML import *
from threeML.io.package_data import get_path_of_data_file
```

```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

import warnings
warnings.filterwarnings('ignore')
```


```python
# Select an astromodels function to from which to simualte
generating_function = Powerlaw(K=1.0, index=-2, piv=10.0)

# set up the x grig points
x_points = np.logspace(0, 2, 50)

# call the from_function classmethod
xyl_generator = XYLike.from_function(
    "sim_data",
    function=generating_function,
    x=x_points,
    yerr=0.3 * generating_function(x_points),
)


xyl_generator.plot(x_scale="log", y_scale="log");
```

#### SpectrumLike

Generating synthetic spectra from SpectrumLike (non-energy dispersed count spectra) can take many forms with different inputs.

First, let's set the energy bins we will use for all generated spectra

```python

energies = np.logspace(0,2,51)

# create the low and high energy bin edges
low_edge = energies[:-1]
high_edge = energies[1:]
```

Now, let's use a blackbody for the source spectrum.

```python
# get a BPL source function
source_function = Blackbody(K=1, kT = 5.)
```

##### Poisson spectrum with no background

```python
spectrum_generator = SpectrumLike.from_function('fake',
                                               source_function=source_function,
                                               energy_min=low_edge,
                                               energy_max=high_edge)


spectrum_generator.view_count_spectrum();
```

##### Gaussian spectrum with no background

```python
spectrum_generator = SpectrumLike.from_function('fake',
                                               source_function=source_function,
                                               source_errors= 0.5 * source_function(low_edge),
                                               energy_min=low_edge,
                                               energy_max=high_edge)


spectrum_generator.view_count_spectrum();
```

##### Poisson spectrum with Poisson Background

```python

# power law background function
background_function = Powerlaw(K=.7,index=-1.5, piv=10.)

spectrum_generator = SpectrumLike.from_function('fake',
                                               source_function=source_function,
                                               background_function=background_function,
                                               energy_min=low_edge,
                                               energy_max=high_edge)


spectrum_generator.view_count_spectrum();
```

##### Poisson spectrum with Gaussian background

```python

spectrum_generator = SpectrumLike.from_function('fake',
                                               source_function=source_function,
                                               background_function=background_function,
                                               background_errors= 0.1 * background_function(low_edge),
                                               energy_min=low_edge,
                                               energy_max=high_edge)


spectrum_generator.view_count_spectrum();
```

#### DispersionSpectrumLike

DispersionSpectrumLike behaves in the same fashion as SpectrumLike except that a 3ML Instrument response must be set which means that the energy bins do not need to be specified as they are derived from the response

Let's grab a response from an instrument.

```python
from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import OGIPResponse

# we will use a demo response
response = OGIPResponse(get_path_of_data_file("datasets/ogip_powerlaw.rsp"))
```

```python tags=["nbsphinx-thumbnail"]
# rescale the functions for the response
source_function = Blackbody(K=1e-7, kT=500.0)
background_function = Powerlaw(K=1, index=-1.5, piv=1.0e3)
spectrum_generator = DispersionSpectrumLike.from_function(
    "fake",
    source_function=source_function,
    background_function=background_function,
    response=response,
)

spectrum_generator.view_count_spectrum();
```

### Generating spectra from fitted models

When performing goodness of fit tests, likelihood ratio tests (both automatic in 3ML) or posterior predictive checks, we need to generate synthetic data from our fitted models. Therefore, we proved methods to do this for most current plugins.


#### XYLike

Let's load some example, generic XY data and fit it with a power law.

```python
data_path = get_path_of_data_file("datasets/xy_powerlaw.txt")


xyl = XYLike.from_text_file("xyl", data_path)

fit_function = Powerlaw()

xyl.fit(fit_function)

xyl.plot(x_scale="log", y_scale="log");
```

Once our fit has been finished, we can produce simulated data sets from those model parameters.

```python
synthetic_xyl = xyl.get_simulated_dataset()

synthetic_xyl.plot(x_scale="log", y_scale="log");
```

#### SpectrumLike and DispersionSpectrumLike (OGIPLike)

Both spectrum plugins work in the same way when generating data from a fit. They both keep track of the statistical properties of the likelihoods in the plugin so that the simulated datasets have the appropriate statistical properties. Additionally, background, responsses, etc. are simulated and/or kept track of as well.

Let's fit an example energy dispersed spectrum.

```python
ogip_data = OGIPLike(
    "ogip",
    observation=get_path_of_data_file("datasets/ogip_powerlaw.pha"),
    background=get_path_of_data_file("datasets/ogip_powerlaw.bak"),
    response=get_path_of_data_file("datasets/ogip_powerlaw.rsp"),
)

ogip_data.view_count_spectrum()
# define the function
fit_function = Cutoff_powerlaw(K=1e-3, xc=1000, index=-0.66)

# define the point source
point_source = PointSource("ps", 0, 0, spectral_shape=fit_function)

# define the model
model = Model(point_source)

ogip_data.set_model(model)
```

Now we can now generate synthetic datasets from the fitted model. This will include the background sampled properly from the profile likelihood. The instrument response is automatically passed to the new plugin.

```python
synthetic_ogip = ogip_data.get_simulated_dataset()

synthetic_ogip.view_count_spectrum();
```
