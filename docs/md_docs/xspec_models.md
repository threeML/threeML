\---
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

<!-- #region -->
## Working with XSPEC models

One of the most powerful aspects of **XSPEC** is a huge modeling community. While in 3ML, we are focused on building a powerful and modular data analysis tool, we cannot neglect the need for many of the models thahat already exist in **XSPEC** and thus provide support for them via **astromodels** directly in 3ML. 

For details on installing **astromodels** with **XSPEC** support, visit the 3ML or **astromodels** installation page. 


Let's explore how we can use **XSPEC** spectral models in 3ML. 
<!-- #endregion -->


```python nbsphinx="hidden"
import warnings
warnings.filterwarnings('ignore')
```

```python
import matplotlib.pyplot as plt
import numpy as np
```

```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)

silence_warnings()

set_threeML_style()
```


We do not load the models by default as this takes some time and 3ML should load quickly. However, if you need the **XSPEC** models, they are imported from astromodels like this:

```python
from astromodels.xspec.factory import *
```

The models are indexed with *XS_* before the typical **XSPEC** model names.

```python
plaw = XS_powerlaw()
phabs = XS_phabs()
phabs

```

The spectral models behave just as any other **astromodels** spectral model and can be used in combination with other **astromodels** spectral models.

```python
from astromodels import Powerlaw

am_plaw = Powerlaw()

plaw_with_abs = am_plaw*phabs


fig, ax =plt.subplots()

energy_grid = np.linspace(.1,10.,1000)

ax.loglog(energy_grid,plaw_with_abs(energy_grid))
ax.set_xlabel('energy')
ax.set_ylabel('flux')

```

## XSPEC Settings

Many **XSPEC** models depend on external abundances, cross-sections, and cosmological parameters. We provide an interface to control these directly.

Simply import the **XSPEC** settings like so:

```python
from astromodels.xspec.xspec_settings import *
```

Calling the functions without arguments simply returns their current settings

```python
xspec_abund()
```

```python
xspec_xsect()
```

```python
xspec_cosmo()
```

To change the settings for abundance and cross-section, provide strings with the normal **XSPEC** naming conventions.

```python
xspec_abund('wilm')
xspec_abund()
```

```python
xspec_xsect('bcmc')
xspec_xsect()
```

To alter the cosmological parameters, one passes either the parameters that should be changed, or all three:

```python
xspec_cosmo(H0=68.)
xspec_cosmo()
```

```python
xspec_cosmo(H0=68.,q0=.1,lambda_0=70.)
xspec_cosmo()
```
