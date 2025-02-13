---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fermi-LAT via FermiPyLike


In this Example we show how to use the fermipy plugin in threeML. We perform a Binned likelihood analysis and a Bayesian analysis of the Crab, optimizing the parameters of the Crab Pulsar (PSR J0534+2200) keeping fixed the parameters of the Crab Nebula. In the model, the nebula is described by two sources, one representing the synchrotron spectrum, the othet the Inverse Compton emission.
In this example we show how to download Fermi-LAT data, how to build a model starting from the 4FGL, how to free and fix parameters of the sources in the model, and how to perform a spectral analysis using the fermipy plugin.

```python
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")
import shutil
from IPython.display import Image,display
import glob
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits as pyfits
import scipy as sp

```


```python
%%capture
from threeML import *

```


```python
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

```



## The Fermi 4FGL catalog
Let's interrogate the 4FGL to get the sources in a radius of 20.0 deg around the Crab

```python
lat_catalog = FermiLATSourceCatalog()

ra, dec, table = lat_catalog.search_around_source("Crab", radius=20.0)

table
```

This gets a 3ML model (a Model instance) from the table above, where every source in the 4FGL becomes a Source instance. Note that by default all parameters of all sources are fixed.

```python
model = lat_catalog.get_model()
```

Let's free all the normalizations within 3 deg from the center.

```python
model.free_point_sources_within_radius(3.0, normalization_only=True)

model.display()
```

but then let's fix the sync and the IC components of the Crab nebula (cannot fit them with just one month of data) (these two methods are equivalent)

```python
model['Crab_IC.spectrum.main.Log_parabola.K'].fix = True
model.Crab_synch.spectrum.main.Log_parabola.K.fix     = True
```

However, let's free the index of the Crab Pulsar

```python
model.PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index.free = True

model.display()
```

```python
# Download data from Jan 01 2010 to February 1 2010

tstart = "2010-01-01 00:00:00"
tstop  = "2010-02-01 00:00:00"

# Note that this will understand if you already download these files, and will
# not do it twice unless you change your selection or the outdir

evfile, scfile = download_LAT_data(
    ra,
    dec,
    20.0,
    tstart,
    tstop,
    time_type="Gregorian",
    destination_directory="Crab_data",
)
```

## Configuration for Fermipy

3ML provides and intreface into [Fermipy](https://fermipy.readthedocs.io/en/latest/) via the **FermipyLike** plugin. We can use it to generate basic configuration files.



.. note::
    Currently, the FermipyLike plugin does not provide an interface to handle extended sources. This will change


```python
config = FermipyLike.get_basic_config(evfile=evfile, scfile=scfile, ra=ra, dec=dec, fermipy_verbosity = 1, fermitools_chatter = 0)

# See what we just got

config.display()
```

You can of course modify the configuration as a dictionary

```python
config["selection"]["emax"] = 300000.0
```

and even add sections

```python
config["gtlike"] = {"edisp": False}

config.display()
```

### FermipyLike
Let's create an instance of the plugin/ Note that here no processing is made, because fermipy still doesn't know about the model you want to use.



```python
LAT = FermipyLike("LAT", config)
```

The plugin modifies the configuration as needed to get the output files in a unique place, which will stay the same as long as your selection does not change.

```python
config.display()
```

Here is where the fermipy processing happens (the .setup method)

```python
fermipy_output_directory = Path(config['fileio']['outdir'])
print('Fermipy Output directoty: %s' % fermipy_output_directory)

#This remove the output directory, to start a fresh analysis...

if fermipy_output_directory.exists():  shutil.rmtree(fermipy_output_directory)

# Here is where the fermipy processing happens (the .setup method)

data = DataList(LAT)

jl = JointLikelihood(model, data)
```

The normalization factors of the LAT background components are included in the models as nuisance parameters. They are only added during the previous step (during the model assignment). Let's display them:

```python
for k, v in LAT.nuisance_parameters.items():
    print (k, ":", v)
```

We will fix the isotropic BG as we are not sensitive to it with this dataset. We will also fix one more weak source.

```python
model.LAT_isodiff_Normalization.fix = True
model.x4FGL_J0544d4p2238.spectrum.main.Powerlaw.K.fix = True
model.display()
```

### Performing the fit

```python
jl.set_minimizer("minuit")

res = jl.fit()
```

Now let's compute the errors on the best fit parameters



```python
res = jl.get_errors()
```

We might also want to look at the profile of the likelihood for each parameter.

```python
res = jl.get_contours(
    model.PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index,-2.0,-1.6,30
)
```

```python
res[-1]
```

Or we might want to produce a contour plot

```python
res = jl.get_contours(
    'PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.K',2.1e-13,2.7e-13, 20,
    'PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index',-2.0,-1.7, 20
)
```

```python tags=["nbsphinx-thumbnail"]
res[-1]
```

**Pro-trick:** We can also axcess the GTAnalysis object of fermipy:

```python
#res = jl.fit()
#LAT.gta.write_roi('test',make_plots=True)
```

All the plots are saved in the output directory as png files:



```python
#pngs=Path(f"{fermipy_output_directory}").glob("*png")
#for png in pngs:
#    print(png)
#    my_image=Image(str(png))
#    display(my_image)
```

We can also plot the resulting model:

```python
energies=sp.logspace(1,6,100) *u.MeV
fig, ax=plt.subplots()
# we only want to visualize the relevant sources...
src_to_plot=['Crab','PSR_J0534p2200']
# Now loop over all point sources and plot them
for source_name, point_source in model.point_sources.items():
    for src in src_to_plot: 
        if src in source_name: 
            # Plot the sum of all components for this source

            ax.loglog(energies, point_source(energies),label=source_name)
            # If there is more than one component, plot them also separately

            if len(point_source.components) > 1:

                for component_name, component in point_source.components.items():
                    ax.loglog(energies,component.shape(energies),
                              '--',label=f"{component_name} of {source_name}")
    
# Add a legend
ax.legend(loc=0,frameon=False)

ax.set_xlabel("Energy (MeV)")
ax.set_ylabel(r"Flux (ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$")
ax.set_ylim([1e-18,1e-8])

#show the plot
fig
```

We can also do a bayesian analysis.


This will set priors based on the current defined min-max (log-uniform or uniform). 

```python
for param in model.free_parameters.values():

    if param.has_transformation():
        param.set_uninformative_prior( Log_uniform_prior )
    else:
        param.set_uninformative_prior( Uniform_prior )
```

```python
#It's better to remove the output directory,...
shutil.rmtree(fermipy_output_directory)

bayes = BayesianAnalysis(model, data)
```

Take care of the nuisance parameters `LAT_isodiff_Normalization` and `LAT_galdiff_Prefactor`, which are only created during the previous step.

```python
model.LAT_isodiff_Normalization.fix = True
model.LAT_galdiff_Prefactor.set_uninformative_prior( Log_uniform_prior )
```

```python
bayes.set_sampler('emcee')

n_walkers = 10
burn_in = 10
n_samples = 500

bayes.sampler.setup(n_iterations=n_samples,n_burn_in=burn_in,n_walkers=n_walkers )

res = bayes.sample()

```


You can access to the parameter range like this (HPD):

```python
this_K = bayes.results.get_variates(
    'PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.K'
)
this_idx = bayes.results.get_variates(
    'PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index'
)

print('Highest_posterior_density_intervals :')
print('K (68%%):     %10.2e,%10.2e' % this_K.highest_posterior_density_interval(cl=.68))
print('K (95%%):     %10.2e,%10.2e' % this_K.highest_posterior_density_interval(cl=.95))
print('Index (68%%): %10.2e,%10.2e' % this_idx.highest_posterior_density_interval(cl=.68))
print('Index (95%%): %10.2e,%10.2e' % this_idx.highest_posterior_density_interval(cl=.95))
```

```python
corner_figure = bayes.results.corner_plot()
corner_figure
```

```python

```
