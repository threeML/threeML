---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: 3ml_dev
    language: python
    name: python3
---

# Joint analysis of GRB 130427A with GBM, LLE and LAT data

```python
import warnings
warnings.simplefilter("ignore")
from threeML import *
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import fits
%matplotlib inline
```

```python
gbm_catalog = FermiGBMBurstCatalog()
source_name = 'GRB130427324'
gbm_catalog.query_sources(source_name)
grb_info = gbm_catalog.get_detector_information()[source_name]
grb_info
```

```python
trigger_name = 'bn130427324'
met = 388741629.420
ra = 173.136
dec = 27.7129
tstart = 0
tstop = 100
gbm_detectors = ['n6', 'n9', 'b1']
roi = 10
zmax = 100.0
thetamax = 180.0
irf = "p8_transient010e"
```

## Prepare GBM data

```python
dload = download_GBM_trigger_data(trigger_name, detectors=gbm_detectors)
```

```python
bkg_selection = ["-285--41","310-355"]
gbm_plugins = []
time_series = {}
for det in gbm_detectors:
    
    # Use CSPEC data to fit the background using the background selections. 
    # We use CSPEC because it has a longer duration for fitting the background.
    
    ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
        det, cspec_or_ctime_file=dload[det]["cspec"], rsp_file=dload[det]["rsp"]
    )

    ts_cspec.set_background_interval(*bkg_selection)
    # The background is saved to an HDF5 file that stores the polynomial coefficients and selections
    ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)

    # We use TTE data for the actual spectral analysis
    ts_tte = TimeSeriesBuilder.from_gbm_tte(
        det,
        tte_file=dload[det]["tte"],
        rsp_file=dload[det]["rsp"],
        restore_background=f"{det}_bkg.h5",
    )

    time_series[det] = ts_tte

    # The source selection from the catalog is set
    ts_tte.set_active_time_interval("%f-%f" % (tstart, tstop))

    # The plugin for the time integrated analysis is created for each detector
    fluence_plugin = ts_tte.to_spectrumlike()

    # GBM channel selections for spectral analysis
    if det.startswith("b"):
        fluence_plugin.set_active_measurements("900-30000")

    else:
        fluence_plugin.set_active_measurements("40-900")

    fluence_plugin.rebin_on_background(1.0)

    gbm_plugins.append(fluence_plugin)
```

```python
time_series['n6'].view_lightcurve(-10, 100);
time_series['n9'].view_lightcurve(-10, 100);
time_series['b1'].view_lightcurve(-10, 100);
```

## Prepare LLE data

```python
lle_dload = download_LLE_trigger_data(trigger_name)
```

```python
bkg_selection = ["-285--41","400-500"]
emin, emax = 30*u.MeV, 100*u.MeV
lle_time_series = TimeSeriesBuilder.from_lat_lle("lat_lle", lle_file=lle_dload["lle"], ft2_file=lle_dload["ft2"], rsp_file=lle_dload["rsp"]
)
lle_time_series.set_background_interval(*bkg_selection)
lle_time_series.save_background("lle_bkg.h5", overwrite=True)

lle_time_series.set_active_time_interval("%d-%d" % (tstart, tstop))

lle_plugin = lle_time_series.to_spectrumlike()
lle_plugin.set_active_measurements("%d-%d" % (emin.to('keV').value, emax.to('keV').value))
lle_plugin.use_effective_area_correction(0.8, 1.2)
```

```python
lle_time_series.view_lightcurve(-100, 500);
```

## Prepare LAT data

```python
LATdataset = LAT_dataset()

LATdataset.make_LAT_dataset(
    ra=ra,
    dec=dec,
    radius=12,
    trigger_time=met,
    tstart=0,
    tstop=1000,
    data_type="Extended",
    destination_directory=".",
    Emin=100.0, #100 MeV
    Emax=100000.0, #100 GeV
)
```

```python
LATdataset.extract_events(roi, zmax, irf)

with fits.open(LATdataset.filt_file) as event_file:
    lat_events = event_file["EVENTS"].data
event_times = lat_events["TIME"] - met

%matplotlib inline
fig, axs = plt.subplots(2, 1, sharex=True)
plt.sca(axs[0])
plt.hist(event_times);
plt.ylabel('Events')
plt.sca(axs[1])
plt.scatter(event_times, lat_events['ENERGY'], marker='o', c=lat_events['ENERGY'], norm='log',
            alpha=0.5, zorder=20);
plt.yscale('log')
plt.ylabel('Energy [MeV]')
plt.xlabel('Time - T0 [s]')
plt.grid(True)
plt.show()
```

```python
analysis_builder = TransientLATDataBuilder(
    LATdataset.grb_name,
    outfile=LATdataset.grb_name,
    roi=roi,
    tstarts="%d" % tstart,
    tstops="%d" % tstop,
    irf=irf,
    zmax=zmax,
    galactic_model="template",
    particle_model="isotr template",
    datarepository=".",
)
df = analysis_builder.display(get=True)
```

```python
LAT_observations = analysis_builder.run(recompute_intervals=True)
LAT_plugin = LAT_observations[0].to_LATLike()
```

## Define the first model to test

```python
band = Band()
band.alpha.prior = Truncated_gaussian(lower_bound=-1.5, upper_bound=1, mu=-1, sigma=0.5)
band.beta.prior = Truncated_gaussian(lower_bound=-5, upper_bound=-1.6, mu=-2, sigma=0.5)
band.xp.prior = Log_normal(mu=2, sigma=1)
band.xp.bounds = (None, None)
band.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
source = PointSource(trigger_name, ra, dec, spectral_shape=band)
band_model = Model(source)
```

## Run the joint fit

```python
datalist_1 = DataList(*gbm_plugins, lle_plugin, LAT_plugin)
jl_1 = JointLikelihood(band_model, datalist_1, verbose=False)
band_model.display()
print(jl_1.data_list.keys())
# This is needed to fix the galactic template to 1 (if needed) 
band_model[LAT_plugin.get_name() + "_GalacticTemplate_Value"].value = 1.0
band_model[LAT_plugin.get_name() + "_GalacticTemplate_Value"].fix = True
band_model[LAT_plugin.get_name() + "_IsotropicTemplate_Normalization"].fix = False
#sbpl_model.display(complete=True)
jl_1.set_minimizer("minuit")
jl_1.fit();
```

```python
display_spectrum_model_counts(jl_1);
```

## Define a second model to test

```python
spectrum = Band() + Powerlaw()
# spectrum.K_1 = 0.1
# spectrum.break_energy_1 = 800
# spectrum.beta_1 = -3
# spectrum.K_2 = 3
source_2 = PointSource("%s_2" % trigger_name, ra, dec, spectral_shape=spectrum)
comp_model = Model(source_2)
comp_model.display()
```

## Run the joint fit again

```python
datalist_2 = DataList(*gbm_plugins, lle_plugin, LAT_plugin)
jl_2 = JointLikelihood(comp_model, datalist_2, verbose=False)
comp_model[LAT_plugin.get_name() + "_GalacticTemplate_Value"].value = 1.0
comp_model[LAT_plugin.get_name() + "_GalacticTemplate_Value"].fix = True
comp_model[LAT_plugin.get_name() + "_IsotropicTemplate_Normalization"].fix = False
jl_2.set_minimizer("minuit")
jl_2.fit();
```

```python tags=["nbsphinx-thumbnail"]
display_spectrum_model_counts(jl_2, show_residuals=True);
```

```python
jl_1.results.get_statistic_measure_frame()
```

```python
jl_2.results.get_statistic_measure_frame()
```

## Plot the two components of the best-fit spectrum 

```python
fig = plot_spectra(jl_2.results,
    ene_min=10 * u.keV,
    ene_max=100 * u.GeV,
    flux_unit="erg2/(cm2 s keV)",
    fit_cmap="viridis",
    contour_cmap="viridis",
    contour_style_kwargs=dict(alpha=0.1),
    use_components=True,
    components_to_use = ["total", "Band", "Powerlaw"],
)
```

## Compute the flux

```python
jl_2.results.get_flux(ene_min=10 * u.keV, ene_max=100 * u.GeV, use_components=True, flux_unit="1/(cm2 s)")
```

```python
jl_2.results.get_flux(ene_min=10 * u.keV, ene_max=100 * u.GeV, use_components=False, flux_unit="1/(cm2 s)")
```
