---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.2"
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# HAL (HAWC Accelerated Likelihood) plugin

```python
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")
```

The High-Altitude Water Cherenkov Observatory ([HAWC](https://www.hawc-observatory.org/)) is a ground-based wide-field TeV gamma-ray observatory located in Mexico, scanning about 2/3 of the northern sky every day. It is sensitive to gamma rays in the energy range from hundreds of GeV to hundreds of TeV. In addition to gamma ray-induced air showers, HAWC also detects signals from cosmic-ray induced air showers, which make up the main (and partially irreducible) background. HAWC uses a forward-folding likelihood analysis, similar to the approach that is used e.g. by many Fermi-LAT analyses. Most of HAWC's data are only available to collaboration members. However, HAWC has released several [partial-sky datasets](https://data.hawc-observatory.org/) to the public, and is committed to releasing more in the future. If you are interested in a particular HAWC dataset, you can find contact information for HAWC working group leaders on the linked webpage.

The HAL (HAWC Accelerated Likelihood) plugin for threeML is provided in a separate python package, `hawc_hal`. Before running this example offline, make sure that the `HAL` package is installed. The `hawc_hal` package has a few dependencies: `uproot, awkward, and hist` which are taken care of when you install `HAL`. It can be installed as follows (skip the first step if you already have threeML/astromodels installed):

<!-- uproot, awkward, hist, mplhep -->
<!-- The HAL (HAWC accelerated likelihood) plugin for threeML is provided in a separate python package, `hawc_hal`. Before running this example offline, make sure that the `HAL` plugin is installed. The `hawc_hal` package needs `root_numpy` as a dependency. It can be installed as follows (skip the first step if you already have threeML/astromodels installed): -->

```
conda create -y --name hal_env -c conda-forge -c threeml numpy scipy matplotlib ipython numba reproject "astromodels>=2" "threeml>=2" root
conda activate hal_env
pip install git+https://github.com/threeml/hawc_hal.git
```

<!-- pip install --no-binary :all: root_numpy -->

## Download

First, download the 507 day Crab dataset and instrument response file from the HAWC webpage.
In case of problems with the code below, the files can be downloaded manually from https://data.hawc-observatory.org/datasets/crab_data/index.php . If the files already exist, the code won't try to overwrite them.

```python
import requests
import shutil
import os

def get_hawc_file( filename, odir = "./", overwrite = False ):

    if overwrite or not os.path.exists( odir + filename):
        url="https://data.hawc-observatory.org/datasets/crab_data/public_data/crab_2017/"

        req = requests.get(url+filename, verify=False,stream=True)
        req.raw.decode_content = True
        with open( odir + filename, 'wb') as f:
            shutil.copyfileobj(req.raw, f)

    return odir + filename


maptree = "HAWC_9bin_507days_crab_data.hd5"
response = "HAWC_9bin_507days_crab_response.hd5"
odir = "./"


maptree = get_hawc_file(maptree, odir)
response = get_hawc_file(response, odir)

assert( os.path.exists(maptree))
assert( os.path.exists(response))

```

Next, we need to define the region of interest (ROI) and instantiate the HAL plugin. The provided data file is a partial map covering 3˚ around the nominal position of the Crab nebula, so we need to make sure we use the same ROI (or a subset) here.

Some parameters of note:

`flat_sky_pixels_size=0.1`: HAWC data is provided in counts binned according to the HealPIX scheme, with NSIDE=1024. To reduce computing time, the HAL plugin performs the convolution of the model with the detector response on a rectangular grid on the plane tangent to the center of the ROI. The convolved counts are finally converted back to HealPIX for the calculation of the likelihood. The parameter `flat_sky_pixel_size` of the `HAL` class controls the size of the grid used in the convolution with the detector response. Larger grid spacing reduced computing time, but can make the results less reliable. The grid spacing should not be significantly larger than the HealPix pixel size. Also note that model features smaller than the grid spacing may not contribute to the convolved source image. The default here is 0.1˚

`hawc.set_active_measurements(1, 9)`: HAWC divides its data in general _analysis bins_, each labeled with a string. The dataset provided here uses 9 bins labeled with integers from 1 to 9 and binned according to the fraction of PMTs hit by a particular shower, which is correlated to the energy of the primary particle. See [Abeysekara et al., 2017](https://iopscience.iop.org/article/10.3847/1538-4357/aa7555/meta) for details about the meaning of the bins. There are two ways to set the 'active' bins (bins to be considered for the fit) in the `HAL` plugin: `set_active_measurement(bin_id_min=1, bin_id_max=9)` (can only be used with numerical bins) and `set_active_measurement(bin_list=[1,2,3,4,5,6,7,8,9])`. Additionally, you can specify the number of transits with the argument `set_transits` in the `HAL` plugin. By default, it reads the number of transits from the the maptree.

```python
%%capture
from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
silence_warnings()
%matplotlib inline
from jupyterthemes import jtplot
jtplot.style(context='talk', fscale=1, ticks=True, grid=False)
set_threeML_style()



# Define the ROI.
ra_crab, dec_crab = 83.63,22.02
data_radius = 3.0 #in degree
model_radius = 8.0 #in degree

roi = HealpixConeROI(data_radius=data_radius,
                     model_radius=model_radius,
                     ra=ra_crab,
                     dec=dec_crab)

# Instance the plugin
hawc = HAL("HAWC",
           maptree,
           response,
           roi,
           flat_sky_pixels_size=0.1,
           set_transits=None)

# Use from bin 1 to bin 9
hawc.set_active_measurements(1, 9)

```

## Exploratory analysis

Next, let's explore the HAWC data.

First, print some information about the ROI and the data:

```python
hawc.display()
```

Next, some plots. We can use the `display_stacked_image()` function to visualize the background-subtracted counts, summed over all (active) bins. This function smoothes the counts for plotting and takes the width of the smoothing kernel as an argument. This width must be non-zero.

First, let's see what the counts look like without (or very little) smoothing:

```python
fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.01)
```

What about smoothing it with a smoothing kernel comparable to the PSF? (Note the change in the color scale!)

```python
fig_smooth = hawc.display_stacked_image(smoothing_kernel_sigma=0.2)
```

Now, let's see how the data change from bin to bin (feel free to play around with the smoothing radius here)!

```python
for bin in [1, 5, 9]:

    #set only one active bin
    hawc.set_active_measurements(bin, bin)

    fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.01)
    fig.suptitle(f"Bin {bin}")

```

<!-- fig.suptitle("Bin {}".format(bin)) -->

Smaller bins have more events overall, but also more background, and a larger PSF.

## Simple model fit

Let's set up a simple one-source model with a log-parabolic spectrum. For now, the source position will be fixed to its nominal position.

```python
# Define model
spectrum = Log_parabola()
source = PointSource("crab", ra=ra_crab, dec=dec_crab, spectral_shape=spectrum)

spectrum.piv = 7 * u.TeV
spectrum.piv.fix = True

spectrum.K = 1e-14 / (u.TeV * u.cm ** 2 * u.s)  # norm (in 1/(TeV cm2 s))
spectrum.K.bounds = (1e-35, 1e-10) / (u.TeV * u.cm ** 2 * u.s)  # without units energies would be in keV

spectrum.alpha = -2.5  # log parabolic alpha (index)
spectrum.alpha.bounds = (-4., 2.)

spectrum.beta = 0  # log parabolic beta (curvature)
spectrum.beta.bounds = (-1., 1.)

model = Model(source)

model.display(complete=True)
```

Next, set up the likelihood object and run the fit. Fit results are written to disk so they can be read in again later.

```python
#make sure to re-set the active bins before the fit
hawc.set_active_measurements(1, 9)
data = DataList(hawc)

jl = JointLikelihood(model, data, verbose=False)
jl.set_minimizer("minuit")
param_df, like_df = jl.fit()

results=jl.results
results.write_to("crab_lp_public_results.fits", overwrite=True)
results.optimized_model.save("crab_fit.yml", overwrite=True)

```

## Assessing the Fit Quality

`HAL` and `threeML` provide several ways to assess the quality of the provided model, both quantitatively and qualitatively.

For a first visual impression, we can display the model, excess (counts - background), background, and residuals (counts - model - background) for each bin.

```python
fig = hawc.display_fit(smoothing_kernel_sigma=0.01,display_colorbar=True)
```

Same, but with smoothing:

```python tags=["nbsphinx-thumbnail"]
fig = hawc.display_fit(smoothing_kernel_sigma=0.2,display_colorbar=True)
```

The following plot provides a summary of the images above:

```python
fig = hawc.display_spectrum()
```

The `GoodnessOfFit` module provides a numerical assessment of the goodness of fit, by comparing the best-fit likelihood from the fit to likelihood values seen in "simulated" data (background + source model + pseudo-random poissonian fluctuation).

```python
gof_obj = GoodnessOfFit(jl)
gof, data_frame, like_data_frame = gof_obj.by_mc(n_iterations=200)
```

```python
p = gof["total"]

print("Goodness of fit (p-value:)", p)
print(f"Meaning that {100*p:.1f}% of simulations have a larger (worse) likelihood")
print(f"and {100*(1-p):.1f}% of simulations have a smaller (better) likelihood than seen in data")

df = like_data_frame.reset_index()
df = df[df.level_1 == "total"]

fig, ax = plt.subplots()
ax.hist(df["-log(likelihood)"], label = "fluctuated source models", bins=range(18000,19000,50))
ax.axvline(like_df.loc["total","-log(likelihood)"], label = "fit to data", color = "orange" )
ax.xlabel = "-log(likelihood)"
ax.legend(loc="best")
```

<!-- print("and {:.1f}% of simulations have a smaller (better) likelihood than seen in data".format(100*(1-p) ) ) -->
<!-- print("Meaning that {:.1f}% of simulations have a larger (worse) likelihood".format(100*p) ) -->

Not too bad, but not great 🙃. The model might be improved by freeing the position or choosing another spectral shape. However, note that the detector response file provided here is the one used in the 2017 publication, which has since been superceeded by an updated detector model (only available inside the HAWC collaboration for now). Discrepancies between the simulated and true detector response may also worsen the goodness of fit.

## Visualizing the Fit Results

Now that we have satisfied ourselves that the best-fit model describes the data reasonably well, let's look at the fit parameters. First, let's print them again:

```python
param_df
```

Plot the spectrum using `threeML`'s `plot_spectra` convenience function:

```python
fig, ax = plt.subplots()

plot_spectra(results,
                   ene_min=1.0,
                   ene_max=37,
                   num_ene=50,
                   energy_unit='TeV',
                   flux_unit='TeV/(s cm2)',
                   subplot = ax)
ax.set_xlim(0.8,100)
ax.set_ylabel(r"$E^2\,dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
ax.set_xlabel("Energy [TeV]")
```

What about visualizing the fit parameter's uncertainties and their correlations? We can use _likelihood profiles_ for that. We need to provide the range for the contours. We can derive a reasonable range from the best-fit values and their uncertainties.

```python
range_min = {}
range_max = {}

params = model.free_parameters
N_param = len(model.free_parameters.keys() )

for name in params:
    row = param_df.loc[name]
    range_min[name] = row["value"] + 5*row["negative_error"]
    range_max[name] = row["value"] + 5*row["positive_error"]

for i in range(0,N_param):
    p1 = list(params.keys())[i]
    p2 = list(params.keys())[(i+1)%N_param]

    a, b, cc, fig = jl.get_contours(p1, range_min[p1], range_max[p1], 20,
                                p2, range_min[p2], range_max[p2], 20 )
```

We can see that the index and normalization are essentially uncorrelated (as the pivot energy was chosen accordingly). There is significant correlation between the spectral index and the curvature parameter though.
