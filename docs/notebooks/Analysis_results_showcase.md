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

# Analysis Results

3ML stores the results of a fit in a container we call an "Analysis Result" (AR). The structure of this object is designed to be useable in a *live* sense within an *active* analysis (python script, ipython interactive shell, jupyter notebook) as well as storable as a FITS file for saving results for later.

The structure is nearly the same between MLE and Bayesian analyses in order to make a seamless functionality between all analyses.


```python
from threeML import *

from threeML.analysis_results import *

from tqdm.auto import tqdm

from jupyterthemes import jtplot

%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)


import matplotlib.pyplot as plt

plt.style.use("./threeml.mplstyle")


import astropy.units as u
```

Let's take a look at what we can do with an AR. First, we will simulate some data.

```python
gen_function = Line(a=2, b=0) + Gaussian(F=30.0, mu=25.0, sigma=1)

# Generate a dataset using the line and a gaussian.
# constant 20% error

x = np.linspace(0, 50, 50)

xy = XYLike.from_function(
    "sim_data", function=gen_function, x=x, yerr=0.2 * gen_function(x)
)

xy.plot();
```

<!-- #region -->
## MLE Results


First we will demonstrate how AR's work for an MLE analysis on our synthetic data. As we will see, most of the functionality exists in the Bayesian AR's as well. 

Let's do a simple likelihood maximization of our data and model.
<!-- #endregion -->

```python
fitfun = Line() + Gaussian()

fitfun.b_1.bounds = (-10, 10.0)
fitfun.a_1.bounds = (-100, 100.0)
fitfun.F_2 = 25.0
fitfun.F_2.bounds = (1e-3, 200.0)
fitfun.mu_2 = 25.0
fitfun.mu_2.bounds = (0.0, 100.0)
fitfun.sigma_2.bounds = (1e-3, 10.0)

model = Model(PointSource("fake", 0.0, 0.0, fitfun))

data = DataList(xy)

jl = JointLikelihood(model, DataList(xy))
_ = jl.fit()
```

We can get our errors as always, but the results cannot be propagated (error propagation assumes Gaussian errors, i.e., symmetric errors)
In this case though errors are pretty symmetric, so we are likely in the case
where the MLE is actually normally distributed.

```python
jl.get_errors();
```

We need to get the AnalysisResults object that is created after a fit is performed. The AR object is a member of the JointLikelihood object

```python
ar = jl.results
```

We can display the results of the analysis. Note, when a fit is performed, the post display is actaully from the internal AR.

```python
ar.display()
```

By default, the equal tail intervals are displayed. We can instead display highest posterior densities (equal in the MLE case)

```python
ar.display("hpd")
```

The AR stores several properties from the analysis:

```python
ar.analysis_type
```

```python
ar.covariance_matrix
```

```python
ar.get_point_source_flux(1*u.keV, .1*u.MeV)
```

```python
ar.optimized_model
```

## Saving results to disk

The beauty of the analysis result is that all of this information can be written to disk and restored at a later time. The statistical parameters, best-fit model, etc. can all be recovered.

AR's are stored as a structured FITS file. We write the AR like this:

```python
ar.write_to("test_mle.fits", overwrite=True)
```

The FITS file can be examines with any normal FITS reader.

```python
import astropy.io.fits as fits
```

```python
ar_fits = fits.open('test_mle.fits')
ar_fits.info()
```

However, to easily pull the results back into the 3ML framework, we use the ${\tt load\_analysis\_results}$ function:

```python
ar_reloaded = load_analysis_results("test_mle.fits")
```

```python
ar_reloaded.get_statistic_frame()
```

You can get a DataFrame with the saved results:

```python
ar_reloaded.get_data_frame()
```

## Analysis Result Sets


When doing time-resolved analysis or analysing a several objects, we can save several AR's is a set. This is achieved with the analysis result set. We can pass an array of AR's to the set and even set up descriptions for the different entries.

```python
from threeML.analysis_results import AnalysisResultsSet

analysis_set = AnalysisResultsSet([ar, ar_reloaded])

# index as time bins
analysis_set.set_bins("testing", [-1, 1], [3, 5], unit="s")

# write to disk
analysis_set.write_to("analysis_set_test.fits", overwrite=True)
```

```python
analysis_set = load_analysis_results("analysis_set_test.fits")
```

```python
analysis_set[0].display()
```

## Error propagation
In 3ML, we propagate errors for MLE reults via sampling of the covariance matrix *instead* of Taylor exanding around the maximum of the likelihood and computing a jacobain. Thus, we can achieve non-linear error propagation.

You can use the results for propagating errors non-linearly for analytical functions:


```python
p1 = ar.get_variates("fake.spectrum.main.composite.b_1")
p2 = ar.get_variates("fake.spectrum.main.composite.a_1")

print("Propagating a+b, with a and b respectively:")
print(p1)
print(p2)

print("\nThis is the result (with errors):")
res = p1 + p2
print(res)

print(res.equal_tail_interval())
```

The propagation accounts for covariances. For example this
has error of zero (of course) since there is perfect covariance.

```python
print("\nThis is 50 * a/a:")
print(50 * p1/p1)
```

You can use arbitrary (np) functions

```python
print("\nThis is arcsinh(b + 5*) / np.log10(b) (why not?)")
print(np.arcsinh(p1 + 5 * p2) / np.log10(p2))
```

Errors can become asymmetric. For example, the ratio of two gaussians is
asymmetric notoriously:

```python
print("\nRatio a/b:")
print(p2 / p1)
```

You can always use it with arbitrary functions:

```python
def my_function(x, a, b):

    return b * x ** a


print("\nPropagating using a custom function:")
print(my_function(2.3, p1, p2))
```

This is an example of an error propagation to get the plot of the model with its errors
(which are propagated without assuming linearity on parameters)

```python
def go(fitfun, ar, model):

    fig, ax = plt.subplots()

    # Gather the parameter variates

    arguments = {}

    for par in fitfun.parameters.values():

        if par.free:

            this_name = par.name

            this_variate = ar.get_variates(par.path)

            # Do not use more than 1000 values (would make computation too slow for nothing)

            if len(this_variate) > 1000:

                this_variate = np.random.choice(this_variate, size=1000)

            arguments[this_name] = this_variate

    # Prepare the error propagator function

    pp = ar.propagate(
        ar.optimized_model.fake.spectrum.main.shape.evaluate_at, **arguments
    )

    # You can just use it as:

    print(pp(5.0))

    # Make the plot

    energies = np.linspace(0, 50, 100)

    low_curve = np.zeros_like(energies)
    middle_curve = np.zeros_like(energies)
    hi_curve = np.zeros_like(energies)

    free_parameters = model.free_parameters

    p = tqdm(total=len(energies), desc="Propagating errors")

    with use_astromodels_memoization(False):
        for i, e in enumerate(energies):
            this_flux = pp(e)

            low_bound, hi_bound = this_flux.equal_tail_interval()

            low_curve[i], middle_curve[i], hi_curve[i] = (
                low_bound,
                this_flux.median,
                hi_bound,
            )

            p.update(1)

    ax.plot(energies, middle_curve, "--", color="black")
    ax.fill_between(energies, low_curve, hi_curve, alpha=0.5, color="blue")
```

```python tags=["nbsphinx-thumbnail"]
go(fitfun, ar, model)
```

## Bayesian Analysis Results
Analysis Results work exactly the same under Bayesian analysis. 

Let's run the analysis first.

```python

for parameter in ar.optimized_model:
    
    model[parameter.path].value = parameter.value

model.fake.spectrum.main.composite.a_1.set_uninformative_prior(Uniform_prior)
model.fake.spectrum.main.composite.b_1.set_uninformative_prior(Uniform_prior)
model.fake.spectrum.main.composite.F_2.set_uninformative_prior(Log_uniform_prior)
model.fake.spectrum.main.composite.mu_2.set_uninformative_prior(Uniform_prior)
model.fake.spectrum.main.composite.sigma_2.set_uninformative_prior(Log_uniform_prior)

bs = BayesianAnalysis(model, data)
bs.set_sampler('emcee')
bs.sampler.setup(n_iterations=1000,n_burn_in=100,n_walkers=20 )
samples = bs.sample()
```

Again, we grab the results from the BayesianAnalysis object:

```python
ar2 = bs.results
```

We can write and read the results to/from a file:

```python
ar2.write_to("test_bayes.fits", overwrite=True)
```

```python
ar2_reloaded = load_analysis_results("test_bayes.fits")
```

The AR holds the posterior samples from the analysis. We can see the saved and live reults are the same:

```python
np.allclose(ar2_reloaded.samples, ar2.samples)
```

**NOTE:** *MLE AR's store samples as well. These are the samples from the covariance matrix*

We can examine the marginal distributions of the parameters:

```python
ar2.corner_plot();

```

We can return pandas DataFrames with equal tail or HPD results.

```python
ar2.get_data_frame("equal tail")
```

```python
ar2.get_data_frame("hpd")
```

Error propagation operates the same way. Internally, the process is the same as the MLE results, however, the samples are those of the posterior rather than the (assumed) covariance matrix.

```python
p1 = ar2.get_variates("fake.spectrum.main.composite.b_1")
p2 = ar2.get_variates("fake.spectrum.main.composite.a_1")

print(p1)
print(p2)

res = p1 + p2

print(res)

```

To demonstrate how the two objects (MLE and Bayes) are the same, we see that our plotting function written for the MLE result works on our Bayesian results seamlessly.

```python
go(fitfun, ar2, model)
```

```python

```
