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

# Quickstart

In this simple example we will generate some simulated data, and fit them with 3ML.


Let's start by generating our dataset:


```python
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")
```


```python
%%capture
from threeML import *
```

```python
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
silence_warnings()
set_threeML_style()
```



```python
# Let's generate some data with y = Powerlaw(x)

gen_function = Powerlaw()


# Generate a dataset using the power law, and a
# constant 30% error

x = np.logspace(0, 2, 50)

xyl_generator = XYLike.from_function("sim_data", function = gen_function, 
                                     x = x, 
                                     yerr = 0.3 * gen_function(x))

y = xyl_generator.y
y_err = xyl_generator.yerr
```

We can now fit it easily with 3ML:

```python
fit_function = Powerlaw()

xyl = XYLike("data", x, y, y_err)

results = xyl.fit(fit_function)
```

Plot data and model:

```python tags=["nbsphinx-thumbnail"]
fig = xyl.plot(x_scale='log', y_scale='log')
```

Compute the goodness of fit using Monte Carlo simulations (NOTE: if you repeat this exercise from the beginning many time, you should find that the quantity "gof" is a random number distributed uniformly between 0 and 1. That is the expected result if the model is a good representation of the data)

```python
gof, all_results, all_like_values = xyl.goodness_of_fit()

print("The null-hypothesis probability from simulations is %.2f" % gof['data'])
```

The procedure outlined above works for any distribution for the data (Gaussian or Poisson). In this case we are using Gaussian data, thus the log(likelihood) is just half of a $\chi^2$. We can then also use the $\chi^2$ test, which gives a close result without performing simulations:

```python
import scipy.stats

# Retrieve the likelihood values
like_values = results.get_statistic_frame()

# Compute the number of degrees of freedom
n_dof = len(xyl.x) - len(fit_function.free_parameters)

# Get the observed value for chi2 
# (the factor of 2 comes from the fact that the Gaussian log-likelihood is half of a chi2)
obs_chi2 = 2 * like_values['-log(likelihood)']['data']

theoretical_gof = scipy.stats.chi2(n_dof).sf(obs_chi2)

print("The null-hypothesis probability from theory is %.2f" % theoretical_gof)
```

There are however many settings where a theoretical answer, such as the one provided by the $\chi^2$ test, does not exist. A simple example is a fit where data follow the Poisson statistic. In that case, the MC computation can provide the answer.
