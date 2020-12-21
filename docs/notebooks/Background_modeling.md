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
    display_name: Python 2
    language: python
    name: python2
---

<!-- #region deletable=true editable=true -->
# Background Modeling

When fitting a spectrum with a background, it is invalid to simply subtract off the background if the background is part of the data's generative model [van Dyk et al. (2001)](http://iopscience.iop.org/article/10.1086/318656/meta). Therefore, we are often left with the task of modeling the statistical process of the background along with our source. 

In typical spectral modeling, we find a few common cases when background is involved. If we have total counts ($S_i$) in $i^{\rm th}$ on $N$ bins observed for an exposure of $t_{\rm s}$ and also a measurement of $B_i$ background counts from looking off source for $t_{\rm b}$ seconds, we can then suppose a model for the source rate ($m_i$) and background rate ($b_i$).

**Poisson source with Poisson background**

This is described by a likelihood of the following form:

$$ L = \prod^N_{i=1} \frac{(t_{\rm s}(m_i+b_i))^{S_i} e^{-t_{\rm s}(m_i+b_i)}}{S_i!} \times \frac{(t_{\rm b} b_i)^{B_i} e^{-t_{\rm b}b_i}}{B_i!}  $$

which is a Poisson likelihood for the total model ($m_i +b_i$) conditional on the Poisson distributed background observation. This is the typical case for e.g. aperture x-ray instruments that observe a source region and then a background region. Both observations are Poisson distributed.

**Poisson source with Gaussian background**

This likelihood is similar, but the conditonal background distribution is described by Gaussian:

$$ L = \prod^N_{i=1} \frac{(t_{\rm s}(m_i+b_i))^{S_i} e^{-t_{\rm s}(m_i+b_i)}}{S_i!} \times \frac{1}{\sigma_{b,i}\sqrt{2 \pi}} \exp \left[ \frac{({B_i} - t_{\rm b} b_i)^2} {2 \sigma_{b,i}^2} \right] $$

where the $\sigma_{b,i}$ are the measured errors on $B_i$. This situation occurs e.g. when the background counts are estimated from a fitted model such as time-domain instruments that estimate the background counts from temporal fits to the lightcurve.

In 3ML, we can fit a background model along with the the source model which allows for arbitrarily low background counts (in fact zero) in channels. The alternative is to use profile likelihoods where we first differentiate the likelihood with respect to the background model

$$ \frac{ \partial L}{{\partial b_i}} = 0$$

and solve for the $b_i$ that maximize the likelihood. Both the Poisson and Gaussian background profile likelihoods are described in the [XSPEC statistics guide](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html). This implicitly yields $N$ parameters to the model thus requiring at least one background count per channel. These profile likelihoods are the default Poisson likelihoods in 3ML when a background model is not used with a **SpectrumLike** (and its children, **DispersionSpectrumLike** and **OGIPLike**) plugin.

Let's examine how to handle both cases.

<!-- #endregion -->

```python deletable=true editable=true
from threeML import *

%matplotlib inline

import warnings

warnings.simplefilter('ignore')
```

<!-- #region deletable=true editable=true -->
First we will create an observation where we have a simulated broken power law source spectrum along with an observed background spectrum. The background is a powerl law continuum with a Gaussian line.
<!-- #endregion -->

```python deletable=true editable=true

# create the simulated observation

energies = np.logspace(1,4,151)

low_edge = energies[:-1]
high_edge = energies[1:]

# get a BPL source function
source_function = Broken_powerlaw(K=2,xb=300,piv=300, alpha=0., beta=-3.)

# power law background function
background_function = Powerlaw(K=.5,index=-1.5, piv=100.) + Gaussian(F=50,mu=511,sigma=20)

spectrum_generator = SpectrumLike.from_function('fake',
                                               source_function=source_function,
                                               background_function=background_function,
                                               energy_min=low_edge,
                                               energy_max=high_edge)


spectrum_generator.view_count_spectrum()
```

<!-- #region deletable=true editable=true -->
## Using a profile likelihood

We have very few counts counts in some channels (in fact sometimes zero), but let's assume we do not know the model for the background. In this case, we will use the profile Poisson likelihood.
<!-- #endregion -->

```python deletable=true editable=true
# instance our source spectrum
bpl = Broken_powerlaw(piv=300,xb=500)

# instance a point source
ra, dec = 0,0
ps_src = PointSource('source',ra,dec,spectral_shape=bpl)

# instance the likelihood model
src_model = Model(ps_src)

# pass everything to a joint likelihood object
jl_profile = JointLikelihood(src_model,DataList(spectrum_generator))


# fit the model
_ = jl_profile.fit()

# plot the fit in count space
_ = spectrum_generator.display_model(step=False) 
```

<!-- #region deletable=true editable=true -->
Our fit recovers the simulated parameters. However, we should have binned the spectrum up such that there is at least one background count per spectral bin for the profile to be valid.
<!-- #endregion -->

```python deletable=true editable=true
spectrum_generator.rebin_on_background(1)

spectrum_generator.view_count_spectrum()

_ = jl_profile.fit()

_ = spectrum_generator.display_model(step=False) 
```

<!-- #region deletable=true editable=true -->
## Modeling the background

Now let's try to model the background assuming we know that the background is a power law with a Gaussian line. We can extract a background plugin from the data by passing the original plugin to a classmethod of spectrum like.
<!-- #endregion -->

```python deletable=true editable=true
# extract the background from the spectrum plugin.
# This works for OGIPLike plugins as well, though we could easily also just read 
# in a bakcground PHA
background_plugin = SpectrumLike.from_background('bkg',spectrum_generator)
```

<!-- #region deletable=true editable=true -->
This constructs a new plugin with only the observed background so that we can first model it.
<!-- #endregion -->

```python deletable=true editable=true
background_plugin.view_count_spectrum()
```

<!-- #region deletable=true editable=true -->
We now construct our background model and fit it to the data. Let's assume we know that the line occurs at 511 keV, but we are unsure of its strength an width. We do not need to bin the data up because we are using a simple Poisson likelihood which is valid even when we have zero counts [Cash (1979)](http://adsabs.harvard.edu/abs/1979ApJ...228..939C).
<!-- #endregion -->

```python deletable=true editable=true
# instance the spectrum setting the line's location to 511
bkg_spectrum = Powerlaw(piv=100) +  Gaussian(F=50,mu=511)

# setup model parameters
# fix the line's location
bkg_spectrum.mu_2.fix = True

# nice parameter bounds
bkg_spectrum.K_1.bounds = (1E-4, 10)
bkg_spectrum.F_2.bounds = (0., 1000)
bkg_spectrum.sigma_2.bounds = (2,30)

ps_bkg = PointSource('bkg',0,0,spectral_shape=bkg_spectrum)

bkg_model = Model(ps_bkg)


jl_bkg = JointLikelihood(bkg_model,DataList(background_plugin))


_ = jl_bkg.fit()

_ = background_plugin.display_model(step=False, data_color='#1A68F0', model_color='#FF9700')
```

<!-- #region deletable=true editable=true -->
We now have a model and estimate for the background which we can use when fitting with the source spectrum. We now create a new plugin with just the total observation and pass our background plugin as the background argument.
<!-- #endregion -->

```python deletable=true editable=true
modeled_background_plugin = SpectrumLike('full',
                                         # here we use the original observation
                                         observation=spectrum_generator.observed_spectrum,
                                         # we pass the background plugin as the background!
                                         background=background_plugin)
```

<!-- #region deletable=true editable=true -->
When we look at out count spectrum now, we will see the *predicted* background, rather than the measured one:
<!-- #endregion -->

```python deletable=true editable=true
modeled_background_plugin.view_count_spectrum()
```

<!-- #region deletable=true editable=true -->
Now we simply fit the spectrum as we did in the profiled case. The background plugin's parameters are stored in our new plugin as nuissance parameters:
<!-- #endregion -->

```python deletable=true editable=true
modeled_background_plugin.nuisance_parameters
```

<!-- #region deletable=true editable=true -->
and the fitting engine will use them in the fit. The parameters will still be connected to the background plugin and its model and thus we can free/fix them there as well as set priors on them.
<!-- #endregion -->

```python deletable=true editable=true
# instance the source model... the background plugin has it's model already specified
bpl = Broken_powerlaw(piv=300,xb=500)

bpl.K.bounds = (1E-5,1E1)
bpl.xb.bounds = (1E1,1E4)

ps_src = PointSource('source',0,0,bpl)

src_model = Model(ps_src)


jl_src = JointLikelihood(src_model,DataList(modeled_background_plugin))

_ = jl_src.fit()
```

```python deletable=true editable=true

# over plot the joint background and source fits
fig = modeled_background_plugin.display_model(step=False)

_ = background_plugin.display_model(data_color='#1A68F0', model_color='#FF9700',model_subplot=fig.axes,step=False)
```

```python deletable=true editable=true

```
