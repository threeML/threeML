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

# The 3ML workflow

Generally, an analysis in 3ML is performed in 3 steps:

1. Load the data: one or more datasets are loaded and then listed in a DataList object
2. Define the model: a model for the data is defined by including one or more PointSource, ExtendedSource or ParticleSource instances
3. Perform a likelihood or a Bayesian analysis: the data and the model are used together to perform either a Maximum Likelihood analysis, or a Bayesian analysis


## Loading data

3ML is built around the concept of _plugins_. A plugin is used to load a particular type of data, or the data from a particular instrument. There is a plugin of optical data, one for X-ray data, one for Fermi/LAT data and so on. Plugins instances can be added and removed at the loading stage without changing any other stage of the analysis (but of course, you need to rerun all stages to update the results).

First, let's import 3ML:


```python 
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")

```


```python
%%capture
from threeML import *
import matplotlib.pyplot as plt

```

```python 
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

```

Let's start by loading one dataset, which in the 3ML workflow means creating an instance of the appropriate plugin:

```python
# Get some example data
from threeML.io.package_data import get_path_of_data_file

data_path = get_path_of_data_file("datasets/xy_powerlaw.txt")

# Create an instance of the XYLike plugin, which allows to analyze simple x,y points
# with error bars
xyl = XYLike.from_text_file("xyl", data_path)

# Let's plot it just to see what we have loaded
fig = xyl.plot(x_scale='log', y_scale='log')
```

Now we need to create a DataList object, which in this case contains only one instance:

```python
data = DataList(xyl)
```

The DataList object can receive one or more plugin instances on initialization. So for example, to use two datasets we can simply do:

```python
# Create the second instance, this time of a different type

pha = get_path_of_data_file("datasets/ogip_powerlaw.pha")
bak = get_path_of_data_file("datasets/ogip_powerlaw.bak")
rsp = get_path_of_data_file("datasets/ogip_powerlaw.rsp")

ogip = OGIPLike("ogip", pha, bak, rsp)

# Now use both plugins
data = DataList(xyl, ogip)
```

The DataList object can accept any number of plugins in input.

You can also create a list of plugins, and then create a DataList using the "expansion" feature of the python language ('*'), like this:

```python
# This is equivalent to write data = DataList(xyl, ogip)

my_plugins = [xyl, ogip]
data = DataList(*my_plugins)
```

This is useful if you need to create the list of plugins at runtime, for example looping over many files.


## Define the model

After you have loaded your data, you need to define a model for them. A model is a collection of one or more sources. A source represents an astrophysical reality, like a star, a galaxy, a molecular cloud... There are 3 kinds of sources: PointSource, ExtendedSource and ParticleSource. The latter is used only in special situations. The models are defined using the package astromodels. Here we will only go through the basics. You can find a lot more information here: [astromodels.readthedocs.org](https://astromodels.readthedocs.org)

### Point sources
A point source is characterized by a name, a position, and a spectrum. These are some examples:

```python
# A point source with a power law spectrum

source1_sp = Powerlaw()
source1 = PointSource("source1", ra=23.5, dec=-22.7, spectral_shape=source1_sp)

# Another source with a log-parabolic spectrum plus a power law

source2_sp = Log_parabola() + Powerlaw()
source2 = PointSource("source2", ra=30.5, dec=-27.1, spectral_shape=source2_sp)

# A third source defined in terms of its Galactic latitude and longitude
source3_sp = Cutoff_powerlaw()
source3 = PointSource("source3", l=216.1, b=-74.56, spectral_shape=source3_sp)
```

### Extended sources

An extended source is characterized by its spatial shape and its spectral shape:

```python
# An extended source with a Gaussian shape centered on R.A., Dec = (30.5, -27.1)
# and a sigma of 3.0 degrees
ext1_spatial = Gaussian_on_sphere(lon0=30.5, lat0=-27.1, sigma=3.0)
ext1_spectral = Powerlaw()

ext1 = ExtendedSource("ext1", ext1_spatial, ext1_spectral)

# An extended source with a 3D function 
# (i.e., the function defines both the spatial and the spectral shape)
ext2_spatial = Continuous_injection_diffusion()
ext2 = ExtendedSource("ext2", ext2_spatial)
```

**NOTE**: not all plugins support extended sources. For example, the XYLike plugin we used above do not, as it is meant for data without spatial resolution. 


### Create the likelihood model


Now that we have defined our sources, we can create a model simply as:

```python
model = Model(source1, source2, source3, ext1, ext2)

# We can see a summary of the model like this:
model.display(complete=True)
```

You can easily interact with the model. For example:

```python
# Fix a parameter
model.source1.spectrum.main.Powerlaw.K.fix = True
# or
model.source1.spectrum.main.Powerlaw.K.free = False

# Free it again
model.source1.spectrum.main.Powerlaw.K.free = True
# or
model.source1.spectrum.main.Powerlaw.K.fix = False

# Change the value
model.source1.spectrum.main.Powerlaw.K = 2.3
# or using physical units (need to be compatible with what shown 
# in the table above)
model.source1.spectrum.main.Powerlaw.K = 2.3 * 1 / (u.cm**2 * u.s * u.TeV)

# Change the boundaries for the parameter
model.source1.spectrum.main.Powerlaw.K.bounds = (1e-10, 1.0)
# you can use units here as well, like:
model.source1.spectrum.main.Powerlaw.K.bounds = (1e-5 * 1 / (u.cm**2 * u.s * u.TeV), 
                                                 10.0 * 1 / (u.cm**2 * u.s * u.TeV))

# Link two parameters so that they are forced to have the same value
model.link(model.source2.spectrum.main.composite.K_1,
           model.source1.spectrum.main.Powerlaw.K)

# Link two parameters with a law. The parameters of the law become free
# parameters in the fit. In this case we impose a linear relationship
# between the index of the log-parabolic spectrum and the index of the
# powerlaw in source2: index_2 = a * alpha_1 + b. 

law = Line()
model.link(model.source2.spectrum.main.composite.index_2,
           model.source2.spectrum.main.composite.alpha_1,
           law)

# If you want to force them to be in a specific relationship,
# say index_2 = alpha_1 + 1, just fix a and b to the corresponding values,
# after the linking, like:
# model.source2.spectrum.main.composite.index_2.Line.a = 1.0
# model.source2.spectrum.main.composite.index_2.Line.a.fix = True
# model.source2.spectrum.main.composite.index_2.Line.b = 0.0
# model.source2.spectrum.main.composite.index_2.Line.b.fix = True

# Now display() will show the links
model.display(complete=True)
```

Now, for the following steps, let's keep it simple and let's use a single point source:

```python
new_model = Model(source1)

source1_sp.K.bounds = (0.01, 100)
```

A model can be saved to disk, and reloaded from disk, as:

```python
new_model.save("new_model.yml", overwrite=True)

new_model_reloaded = load_model("new_model.yml")
```

The output is in [YAML format](http://www.yaml.org/start.html), a human-readable text-based format.


## Perform the analysis

### Maximum likelihood analysis

Now that we have the data and the model, we can perform an analysis very easily:

```python
data = DataList(ogip)

jl = JointLikelihood(new_model, data)

best_fit_parameters, likelihood_values = jl.fit()
```

The output of the fit() method of the JointLikelihood object consists of two pandas DataFrame objects, which can be queried, saved to disk, reloaded and so on. Refer to the [pandas manual](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) for details.

After the fit the JointLikelihood instance will have a .results attribute which contains the results of the fit.

```python
jl.results.display()
```

This object can be saved to disk in a FITS file:

```python
jl.results.write_to("my_results.fits", overwrite=True)
```

The produced FITS file contains the complete definition of the model and of the results, so it can be reloaded in a separate session as:

```python
results_reloaded = load_analysis_results("my_results.fits")

results_reloaded.display()
```

The flux of the source can be computed from the 'results' object (even in another session by reloading the FITS file), as:

```python
fluxes = jl.results.get_flux(100 * u.keV, 1 * u.MeV)

# Same results would be obtained with
# fluxes = results_reloaded.get_point_source_flux(100 * u.keV, 1 * u.MeV)
```

We can change the energy range on the fly... even from the reloaded fit!

```python
fluxes = jl.results.get_flux(100 * u.eV, 1 * u.TeV)
```

We can also plot the spectrum with its error region, as:

```python
fig = plot_spectra(jl.results, ene_min=0.1, ene_max=1e6, num_ene=500, 
                          flux_unit='erg / (cm2 s)')
```

### Bayesian analysis
In a very similar way, we can also perform a Bayesian analysis. As a first step, we need to define the priors for all parameters:

```python


# It can be set using the currently defined boundaries
new_model.source1.spectrum.main.Powerlaw.index.set_uninformative_prior(Uniform_prior)

# or uniform prior can be defined directly, like:
new_model.source1.spectrum.main.Powerlaw.index.prior = Uniform_prior(lower_bound=-3, 
                                                                     upper_bound=0)


# The same for the Log_uniform prior
new_model.source1.spectrum.main.Powerlaw.K.prior = Log_uniform_prior(lower_bound=1e-3, 
                                                                     upper_bound=100)
# or
new_model.source1.spectrum.main.Powerlaw.K.set_uninformative_prior(Log_uniform_prior)

new_model.display(complete=True)
```

Then, we can perform our Bayesian analysis like:

```python
bs = BayesianAnalysis(new_model, data)
bs.set_sampler('ultranest')
bs.sampler.setup()
# This uses the ultranest sampler
samples = bs.sample(quiet=True)
```

The BayesianAnalysis object will now have a "results" member which will work exactly the same as explained for the Maximum Likelihood analysis (see above):

```python
bs.results.display()
```

```python
fluxes_bs = bs.results.get_flux(100 * u.keV, 1 * u.MeV)
```

```python tags=["nbsphinx-thumbnail"]
fig = plot_spectra(bs.results, ene_min=0.1, ene_max=1e6, num_ene=500, 
                          flux_unit='erg / (cm2 s)')
```

We can also produce easily a "corner plot", like:

```python
bs.results.corner_plot();
```

```python

```
