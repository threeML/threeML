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
silence_warnings()
set_threeML_style()
import warnings
warnings.simplefilter('ignore')
```

# Constructing plugins from TimeSeries

Many times we encounter event lists or sets of spectral histograms from which we would like to derive a single or set of plugins. For this purpose, we provide the **TimeSeriesBuilder** which provides a unified interface to time series data. Here we will demonstrate how to construct plugins from different data types.

## Constructing time series objects from different data types

The **TimeSeriesBuilder** currently supports reading of the following data type:
* A generic PHAII data file
* GBM TTE/CSPEC/CTIME files
* LAT LLE files

If you would like to build a time series from your own custom data, consider creating a TimeSeriesBuilder.from_your_data() class method.

### GBM Data 

Building plugins from GBM is achieved in the following fashion

```python
cspec_file = get_path_of_data_file('datasets/glg_cspec_n3_bn080916009_v01.pha')
tte_file = get_path_of_data_file('datasets/glg_tte_n3_bn080916009_v01.fit.gz')
gbm_rsp = get_path_of_data_file('datasets/glg_cspec_n3_bn080916009_v00.rsp2')


gbm_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime('nai3_cspec',
                                                      cspec_or_ctime_file=cspec_file,
                                                      rsp_file=gbm_rsp)

gbm_tte = TimeSeriesBuilder.from_gbm_tte('nai3_tte',
                                          tte_file=tte_file,
                                          rsp_file=gbm_rsp)
```

### LAT LLE data

LAT LLE data is constructed in a similar fashion

```python
lle_file = get_path_of_data_file('datasets/gll_lle_bn080916009_v10.fit')
ft2_file = get_path_of_data_file('datasets/gll_pt_bn080916009_v10.fit')
lle_rsp = get_path_of_data_file('datasets/gll_cspec_bn080916009_v10.rsp')

lat_lle = TimeSeriesBuilder.from_lat_lle('lat_lle',
                                        lle_file=lle_file,
                                        ft2_file=ft2_file,
                                        rsp_file=lle_rsp)
```

## Viewing Lightcurves and selecting source intervals

All time series objects share the same commands to get you to a plugin. 
Let's have a look at the GBM TTE lightcurve.

```python
fig = gbm_tte.view_lightcurve(start=-20,stop=200)
```

Perhaps we want to fit the time interval from 0-10 seconds. We make a selection like this:

```python

gbm_tte.set_active_time_interval('0-10')
fig = gbm_tte.view_lightcurve(start=-20,stop=200);
```

For event list style data like time tagged events, the selection is *exact*. However, pre-binned data in the form of e.g. PHAII files will have the selection automatically adjusted to the underlying temporal bins.

Several discontinuous time selections can be made.

## Fitting a polynomial background

In order to get to a plugin, we need to model and create an estimated background in each channel ($B_i$) for our interval of interest. The process that we have implemented is to fit temporal off-source regions to polynomials ($P(t;\vec{\theta})$) in time. First, a polynomial is fit to the total count rate. From this fit we determine the best polynomial order via a likelihood ratio test, unless the user supplies a polynomial order in the constructor or directly via the polynomial_order attribute. Then, this order of polynomial is fit to every channel in the data.

From the polynomial fit, the polynomial is integrated in time over the active source interval to estimate the count rate in each channel. The estimated background and background errors then stored for each channel.

$$ B_i = \int_{T_1}^{T_2}P(t;\vec{\theta}) {\rm d}t $$


```python
gbm_tte.set_background_interval('-24--5','100-200')
gbm_tte.view_lightcurve(start=-20,stop=200);
```

For event list data, binned or unbinned background fits are possible. For pre-binned data, only a binned fit is possible. 

```python
gbm_tte.set_background_interval('-24--5','100-200',unbinned=False)
```

## Saving the background fit

The background polynomial coefficients can be saved to disk for faster manipulation of time series data.


```python
gbm_tte.save_background('background_store',overwrite=True)
```

```python
gbm_tte_reloaded = TimeSeriesBuilder.from_gbm_tte('nai3_tte',
                                          tte_file=tte_file,
                                          rsp_file=gbm_rsp,
                                          restore_background='background_store.h5')
```

```python
fig = gbm_tte_reloaded.view_lightcurve(-10,200)
```

## Creating a plugin

With our background selections made, we can now create a plugin instance. In the case of GBM data, this results in a **DispersionSpectrumLike**
plugin. Please refer to the Plugins documentation for more details.

```python
gbm_plugin = gbm_tte.to_spectrumlike()
```

```python
gbm_plugin.display()
```

## Time-resolved binning and plugin creation

It is possible to temporally bin time series. There are up to four methods provided depending on the type of time series being used:

* Constant cadence (all time series)
* Custom (all time series)
* Significance (all time series)
* Bayesian Blocks (event lists)


### Constant Cadence

Constant cadence bins are defined by a start and a stop time along with a time delta.


```python
gbm_tte.create_time_bins(start=0, stop=10, method='constant', dt=2.)
```

```python
gbm_tte.bins.display()
```

### Custom

Custom time bins can be created by providing a contiguous list of start and stop times.



```python
time_edges = np.array([.5,.63,20.,21.])

starts = time_edges[:-1]

stops = time_edges[1:]

gbm_tte.create_time_bins(start=starts, stop=stops, method='custom')
```

```python
gbm_tte.bins.display()
```

### Significance

Time bins can be created by specifying a significance of signal to background if a background fit has been performed.

```python
gbm_tte.create_time_bins(start=0., stop=50., method='significance', sigma=25)
```

```python
gbm_tte.bins.display()
```

### Bayesian Blocks

The Bayesian Blocks algorithm (Scargle et al. 2013) can be used to bin event list by looking for significant changes in the rate. 


```python
gbm_tte.create_time_bins(start=0., stop=50., method='bayesblocks', p0=.01, use_background=True)
```

```python
gbm_tte.bins.display()
```

### Working with bins

The light curve can be displayed by supplying the use_binner option to display the time binning


```python
fig = gbm_tte.view_lightcurve(use_binner=True)
```

The bins can all be writted to a PHAII file for analysis via OGIPLike.

```python
gbm_tte.write_pha_from_binner(file_name='out', overwrite=True,
                              force_rsp_write = False)  # if you need to write the RSP to a file. We try to choose the best option for you.
```

Similarly, we can create a list of plugins directly from the time series.

```python
my_plugins = gbm_tte.to_spectrumlike(from_bins=True)
```

```python

```
