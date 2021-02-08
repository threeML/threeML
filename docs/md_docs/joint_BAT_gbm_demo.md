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

# Example joint fit between GBM and Swift BAT

One of the key features of 3ML is the abil ity to fit multi-messenger data properly. A simple example of this is the joint fitting of two instruments whose data obey different likelihoods. Here, we have GBM data which obey a Poisson-Gaussian profile likelihoog (<a href=http://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/node293.html> PGSTAT</a> in XSPEC lingo) and Swift BAT which data which are the result of a "fit" via a coded mask and hence obey a Gaussian ( $\chi^2$ ) likelihood.


```python nbsphinx="hidden"
import warnings
warnings.simplefilter('ignore')

```


```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12345)
from threeML import *
from threeML.io.package_data import get_path_of_data_file
from threeML.io.logging import silence_console_log


```


```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

```



## Plugin setup

We have data from the same time interval from Swift BAT and a GBM NAI and BGO detector. We have preprocessed GBM data to so that it is OGIP compliant. (Remember that we can handle the raw data with the TimeSeriesBuilder). Thus, we will use the OGIPLike plugin to read in each dataset, make energy selections and examine the raw count spectra. 



### Swift BAT

```python
bat_pha = get_path_of_data_file("datasets/bat/gbm_bat_joint_BAT.pha")
bat_rsp = get_path_of_data_file("datasets/bat/gbm_bat_joint_BAT.rsp")

bat = OGIPLike("BAT", observation=bat_pha, response=bat_rsp)

bat.set_active_measurements("15-150")
bat.view_count_spectrum()
```

### Fermi GBM

```python
nai6 = OGIPLike(
    "n6",
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_NAI_06.pha"),
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_NAI_06.bak"),
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_NAI_06.rsp"),
    spectrum_number=1,
)


nai6.set_active_measurements("8-900")
nai6.view_count_spectrum()

bgo0 = OGIPLike(
    "b0",
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_BGO_00.pha"),
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_BGO_00.bak"),
    get_path_of_data_file("datasets/gbm/gbm_bat_joint_BGO_00.rsp"),
    spectrum_number=1,
)

bgo0.set_active_measurements("250-30000")
bgo0.view_count_spectrum()
```

## Model setup

We setup up or spectrum and likelihood model and combine the data. 3ML will automatically assign the proper likelihood to each data set. At first, we will assume a perfect calibration between the different detectors and not a apply a so-called effective area correction. 

```python
band = Band()

model_no_eac = Model(PointSource("joint_fit_no_eac", 0, 0, spectral_shape=band))
```

## Spectral fitting

Now we simply fit the data by building the data list, creating the joint likelihood and running the fit.


### No effective area correction

```python
data_list = DataList(bat, nai6, bgo0)

jl_no_eac = JointLikelihood(model_no_eac, data_list)

jl_no_eac.fit();
```

The fit has resulted in a very typical Band function fit. Let's look in count space at how good of a fit we have obtained.


```python
threeML_config.plugins.ogip.fit_plot.model_cmap = "Set1"
threeML_config.plugins.ogip.fit_plot.n_colors = 3
display_spectrum_model_counts(
    jl_no_eac, 
    min_rate=[0.01, 10.0, 10.0], data_colors=["grey", "k", "k"], 
    show_background=False,
    source_only=True
);
```

It seems that the effective areas between GBM and BAT do not agree! We can look at the goodness of fit for the various data sets.

```python
gof_object = GoodnessOfFit(jl_no_eac)

gof, res_frame, lh_frame = gof_object.by_mc(n_iterations=100)
```

```python
import pandas as pd
pd.Series(gof)
```

Both the GBM NaI detector and Swift BAT exhibit poor GOF.


### With effective are correction

Now let's add an effective area correction between the detectors to see if this fixes the problem. The effective area is a nuissance parameter that attempts to model systematic problems in a instruments calibration. It simply scales the counts of an instrument by a multiplicative factor. It cannot handle more complicated energy dependent 

```python
# turn on the effective area correction and set it's bounds
nai6.use_effective_area_correction(0.2, 1.8)
bgo0.use_effective_area_correction(0.2, 1.8)

model_eac = Model(PointSource("joint_fit_eac", 0, 0, spectral_shape=band))

jl_eac = JointLikelihood(model_eac, data_list)

jl_eac.fit();
```

Now we have a much better fit to all data sets

```python tags=["nbsphinx-thumbnail"]
display_spectrum_model_counts(
    jl_eac, step=False, min_rate=[0.01, 10.0, 10.0], data_colors=["grey", "k", "k"]
);
```

```python
gof_object = GoodnessOfFit(jl_eac)

# for display purposes we are keeping the output clear
# with silence_console_log(and_progress_bars=False):
gof, res_frame, lh_frame = gof_object.by_mc(
    n_iterations=100, continue_on_failure=True )
```

```python
import pandas as pd
pd.Series(gof)
```

## Examining the differences

Let's plot the fits in model space and see how different the resulting models are.


```python
plot_spectra(
    jl_eac.results,
    jl_no_eac.results,
    fit_cmap="Set1",
    contour_cmap="Set1",
    flux_unit="erg2/(keV s cm2)",
    equal_tailed=True,
);
```

We can easily see that the models are different 
