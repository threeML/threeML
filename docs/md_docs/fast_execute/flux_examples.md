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

<!-- #region -->
# Point Source Fluxes and Multiple Sources

Once one has computed a spectral to a point source, getting the flux of that source is possible.
In 3ML, we can obtain flux in a variety of units in a live analysis or from saved fits. There is no need to know exactly what you want to obtain at the time you do the fit.

Also, let's explore how to deal with fitting multiple point sources and linking of parameters.


Let's explore the possibilites.


<!-- #endregion -->

```python
import warnings
warnings.simplefilter("ignore")

```

```python
%%capture
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import astropy.units as u
from threeML import *
from threeML.utils.OGIP.response import OGIPResponse
from threeML.io.package_data import get_path_of_data_file

```

```python
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
silence_warnings()
set_threeML_style()
```

<!-- #region -->
## Generating some synthetic data

![alt text](http://aasnova.org/wp-content/uploads/2016/03/fig16.jpg)

Let's say we have two galactic x-ray sources, some accreting compact binaries perhaps? We observe them at two different times. These sources (imaginary) sources emit a blackbody which is theorized to always be at the same temperature, but perhaps at different flux levels.


Lets simulate one of these sources:

<!-- #endregion -->

```python
np.random.seed(1234)

# we will use a demo response
response_1 = OGIPResponse(get_path_of_data_file("datasets/ogip_powerlaw.rsp"))


source_function_1 = Blackbody(K=5e-8, kT=500.0)
background_function_1 = Powerlaw(K=1, index=-1.5, piv=1.0e3)


spectrum_generator_1 = DispersionSpectrumLike.from_function(
    "s1",
    source_function=source_function_1,
    background_function=background_function_1,
    response=response_1,
)

fig = spectrum_generator_1.view_count_spectrum()
```

Now let's simulate the other source, but this one has an extra feature! There is a power law component in addition to the blackbody. 

```python

response_2 = OGIPResponse(get_path_of_data_file("datasets/ogip_powerlaw.rsp"))


source_function_2 = Blackbody(K=1e-7, kT=500.0) + Powerlaw_flux(F=2e2, index=-1.5, a=10, b=500)
background_function_2 = Powerlaw(K=1, index=-1.5, piv=1.0e3)


spectrum_generator_2 = DispersionSpectrumLike.from_function(
    "s2",
    source_function=source_function_2,
    background_function=background_function_2,
    response=response_2,
)

fig = spectrum_generator_2.view_count_spectrum()
```

## Make the model

Now let's make the model we will use to fit the data. First, let's make the spectral function for source_1 and set priors on the parameters.


```python
spectrum_1 = Blackbody()

spectrum_1.K.prior = Log_normal(mu=np.log(1e-7), sigma=1)
spectrum_1.kT.prior = Log_normal(mu=np.log(300), sigma=2)

ps1 = PointSource("src1", ra=1, dec =20, spectral_shape=spectrum_1)

```

We will do the same for the other source but also include the power law component 

```python
spectrum_2 = Blackbody() + Powerlaw_flux(a=10, b=500) # a,b are the bounds for the flux for this model

spectrum_2.K_1.prior = Log_normal(mu=np.log(1e-6), sigma=1)
spectrum_2.kT_1.prior = Log_normal(mu=np.log(300), sigma=2)

spectrum_2.F_2.prior = Log_normal(mu=np.log(1e2), sigma=1)
spectrum_2.F_2.bounds = (None, None)

spectrum_2.index_2.prior = Gaussian(mu=-1., sigma = 1)
spectrum_2.index_2.bounds = (None, None)

ps2 = PointSource("src2", ra=2, dec=-10, spectral_shape=spectrum_2)
```

Now we can combine these two sources into our model.

```python
model = Model(ps1, ps2)
```

### Linking parameters

We hypothesized that both sources should have the a same blackbody temperature. We can impose this by linking the temperatures. 


```python
model.link(model.src1.spectrum.main.Blackbody.kT, 
           model.src2.spectrum.main.composite.kT_1)
```

we could also link the parameters with an arbitrary function rather than directly. Check out the [astromodels documentation](https://astromodels.readthedocs.io/en/latest/Model_tutorial.html#linking-parameters) for more details.

```python
model
```

### Assigning sources to plugins

Now, if we simply passed out model to the BayesianAnalysis or JointLikelihood objects, it would sum the point source spectra together and apply both sources to all data. 

This is not what we want. Many plugins have the ability to be assigned directly to a source. Let's do that here:

```python
spectrum_generator_1.assign_to_source("src1")

spectrum_generator_2.assign_to_source("src2")
```

Now we simply make our our data list

```python
data = DataList(spectrum_generator_1, spectrum_generator_2)
```

## Fitting the data

Now we fit the data as we normally would. We use Bayesian analysis here.

```python
ba = BayesianAnalysis(model, data)
ba.set_sampler("ultranest")
ba.sampler.setup(frac_remain=0.5)
_ = ba.sample()
```

Let's examine the fits.

```python
fig = display_spectrum_model_counts(ba);
ax = fig.get_axes()[0]
ax.set_ylim(1e-6)
```

Lets grab the result. Remember, we can save the results to disk, so all of the following operations can be run at a later time without having to redo all the above steps!

```python tags=["nbsphinx-thumbnail"]
result = ba.results
fig = result.corner_plot();
```

## Computing fluxes

Now we will compute fluxes. We can compute them an many different units, over any energy range also specified in any units. 

The flux is computed by integrating the function over energy. By default, a fast trapezoid method is used. If you need more accuracy, you can change the method in the configuration.


```python
threeML_config.point_source.integrate_flux_method = "quad"

result.get_flux(ene_min=1*u.keV, 
                ene_max = 1*u.MeV,
                flux_unit="erg/cm2/s")
```

We see that a pandas dataframe is returned with all the information. We could change the confidence region for the uncertainties if we desire. However, we could also sum the source fluxes! 3ML will take care of propagating the uncertainties (for any of these operations). 

```python
threeML_config.point_source.integrate_flux_method = "trapz"

result.get_flux(ene_min=1*u.keV, 
                ene_max = 1*u.MeV,
                confidence_level=0.95,
                sum_sources=True,
                flux_unit="erg/cm2/s")
```

We can get the fluxes of individual components:

```python
result.get_flux(ene_min=10*u.keV, 
                ene_max = 0.5*u.MeV,
                use_components=True,
                flux_unit="1/(cm2 s)")
```

As well as choose which source and component to compute

```python
result.get_flux(ene_min=10*u.keV, 
                ene_max = 0.5*u.MeV,
                sources=["src2"],
                use_components=True,
                components_to_use =["Blackbody"],
                flux_unit="1/(cm2 s)")
```

Finally, the returned flux object is a pandas table and can be manipulated as such:

```python
flux = result.get_flux(ene_min=1*u.keV, 
                ene_max = 1*u.MeV,
                flux_unit="erg/cm2/s")
```

```python
flux["flux"]
```

```python
flux["flux"]["src1: total"]
```
