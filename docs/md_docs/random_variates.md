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

# Random Variates

When we perform a fit or load and analysis result, the parmeters of our model become distributions in the AnalysisResults object. These are actaully instantiactions of the RandomVaraiates class. 

While we have covered most of the functionality of RandomVariates in the AnalysisResults section, we want to highlight a few of the details here.


```python nbsphinx="hidden"
import warnings
warnings.simplefilter('ignore')

```



```python
import matplotlib.pyplot as plt
from threeML import *
```

```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

```

Let's load back our fit of the line + gaussian from the AnalysisResults section.

```python
ar = load_analysis_results('test_mle.fits')
```

When we display our fit, we can see the **parameter paths** of the model. What if we want specific information on a parameter(s)?

```python
ar.display()
```

Let's take a look at the normalization of the gaussian. To access the parameter, we take the parameter path, and we want to get the variates:

```python
norm = ar.get_variates('fake.spectrum.main.composite.F_2')
```

Now, norm is a RandomVariate.

```python
type(norm)
```

This is essentially a wrapper around numpy NDArray with a few added properties. It is an array of samples. In the MLE case, they are samples from the covariance matrix (this is not at all a marginal distribution, but the parameter "knows" about the entire fit, i.e., it is *not* a profile) and in the Bayesian case, these are samples from the posterior (this is a marginal).

The output representation for an RV are its 68% equal-tail and HPD uncertainties.

```python
norm
```

We can access these directly, and to any desired confidence level.

```python
norm.equal_tail_interval(cl=0.95)
```

```python
norm.highest_posterior_density_interval(cl=0.5)
```

As stated above, the RV is made from samples. We can histogram them to show this explicitly.

```python tags=["nbsphinx-thumbnail"]
fig, ax = plt.subplots()

ax.hist(norm.samples,bins=50, ec='k', fc='w', lw=1.2);
ax.set_xlabel('norm');
```

We can easily transform the RV through propagation.

```python
log_norm = np.log10(norm)
log_norm
```

```python
fig, ax = plt.subplots()

ax.hist(log_norm.samples,bins=50,ec='k', fc='w', lw=1.2);
ax.set_xlabel('log norm');
```

<!-- #raw -->
.. note::
    Some operations will destroy the RV by accessing only its NDArray substructure. For example, using an RV with astropy units will return an array of samples with the given units. 
<!-- #endraw -->
