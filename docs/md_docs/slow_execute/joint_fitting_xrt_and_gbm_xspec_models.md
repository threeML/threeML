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

# Joint fitting XRT and GBM data with XSPEC models

### Goals

3ML is designed to properly joint fit data from different instruments with thier instrument dependent likelihoods.
We demostrate this with joint fitting data from GBM and XRT while simultaneously showing hwo to use the XSPEC models form **astromodels**

### Setup

You must have you HEASARC initiated so that **astromodels** can find the XSPEC libraries. 


```python 
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")
```


```python
%%capture
import matplotlib.pyplot as plt
from pathlib import Path
from threeML import *
from threeML.io.package_data import get_path_of_data_file

# we will need XPSEC models for extinction
from astromodels.xspec import *
from astromodels.xspec.xspec_settings import *

```


```python 
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()
```


## Load XRT data

Make a likelihood for the XRT including all the appropriate files

```python
trigger="GRB110731A"
dec=-28.546
ra=280.52

p = Path("datasets/xrt")

xrt = OGIPLike("XRT",observation= get_path_of_data_file(p /"xrt_src.pha"),
                   background= get_path_of_data_file(p /"xrt_bkg.pha"),
                   response= get_path_of_data_file(p /"xrt.rmf"),
                   arf_file= get_path_of_data_file(p /"xrt.arf"))



fig = xrt.view_count_spectrum()
ax = fig.get_axes()[0]
_ =ax.set_xlim(1e-1);
```

```python
fit = xrt.view_count_spectrum(scale_background=False);
```

## Load GBM data

Load all the GBM data you need and make appropriate background, source time, and energy selections. Make sure to check the light curves! 

```python
trigger_number = 'bn110731465'
gbm_data = download_GBM_trigger_data(trigger_number,detectors=['n3'])
```

```python
# Select the time interval
src_selection = "100.169342-150.169342"
bkg_selection = ["-25.0--10.0","300-400"]
ts = TimeSeriesBuilder.from_gbm_cspec_or_ctime(name="gbm_n3",
    cspec_or_ctime_file=gbm_data["n3"]["cspec"],
                                              rsp_file=gbm_data["n3"]["rsp"]
                                              )


ts.set_background_interval(*bkg_selection)
ts.save_background("n3_bkg.h5",overwrite=True)
fig = ts.view_lightcurve(-50,450);
```

```python
ts = TimeSeriesBuilder.from_gbm_tte("gbm_n3",
                                    tte_file=gbm_data["n3"]["tte"],
                                    rsp_file=gbm_data["n3"]["rsp"],
                                    restore_background="n3_bkg.h5"
                                   
                                   )


ts.set_active_time_interval(src_selection)

fig = ts.view_lightcurve(90, 160);
```

```python
nai3 = ts.to_spectrumlike()
```

Make energy selections and check them out

```python
nai3.set_active_measurements("8-900")
fig = nai3.view_count_spectrum();
```

## Setup the model

**astromodels** allows you to use XSPEC models if you have XSPEC installed.
Set all the normal parameters you would in XSPEC and build a model the normal **3ML/astromodels** way! Here we will use the ```phabs``` model from XSPEC and mix it with powerlaw model in astromodels.



### With XSPEC

```python
xspec_abund('angr')

spectral_model =  XS_phabs()* XS_zphabs() * Powerlaw()
spectral_model.set_units(u.keV, 1 / (u.keV * u.cm**2 * u.s))

spectral_model.nh_1=0.101
spectral_model.nh_1.bounds = (None, None)
spectral_model.nh_1.fix = True

spectral_model.nh_2=0.1114424
spectral_model.nh_2.fix = True
spectral_model.nh_2.bounds = (None, None)
spectral_model.redshift_2 = 0.618
spectral_model.redshift_2.fix =True
```

### With astromodels PHABS

We can build the exact same models in native astromodels thanks to **Dominique Eckert**.
Here, there is no extra function for redshifting the absorption model, just pass a redshift.


```python
phabs_local = PhAbs(NH=0.101)
phabs_local.NH.fix=True
phabs_local.redshift.fix=True
phabs_src = PhAbs(NH=0.1114424,redshift=0.618)
phabs_src.NH.fix=True
phabs_src.redshift.fix=True
pl = Powerlaw()
spectral_model_native = phabs_local * phabs_src * pl
```

## Setup the joint likelihood

Create a point source object and model. 

Load the data into a data list and create the joint likelihood



### With XSPEC models
First we will fit with the XSPEC model  

```python
ptsrc = PointSource(trigger,ra,dec,spectral_shape=spectral_model)
model = Model(ptsrc)
```

#### Fitting

```python
data = DataList(xrt,nai3)

jl = JointLikelihood(model, data, verbose=False)
model.display()
```

```python
res = jl.fit()
fig = display_spectrum_model_counts(jl,min_rate=[.5,.1]);
```

```python
res = jl.get_contours(spectral_model.index_3,-2.5,-1.5,50)
```

```python
_ = jl.get_contours(spectral_model.K_3,.1,.3,25,
                spectral_model.index_3,-2.5,-1.5,50);
```

```python
fig = plot_spectra(jl.results,show_legend=False, emin=.01*u.keV);
```

### Fit with astromodels PhAbs

Now lets repeat the fit in pure astromodels.

```python
ptsrc_native = PointSource(trigger,ra,dec,spectral_shape=spectral_model_native)
model_native = Model(ptsrc_native)
```

#### Fitting

```python
data = DataList(xrt,nai3)

jl_native = JointLikelihood(model_native, data, verbose=False)
model.display()
```

```python tags=["nbsphinx-thumbnail"]
res = jl_native.fit()
fig = display_spectrum_model_counts(jl_native,min_rate=[.5,.1]);
```

```python
fig = plot_spectra(jl.results,jl_native.results,show_legend=False, emin=.01*u.keV);
```

Both approaches give the same answer as they should. 

```python

```
