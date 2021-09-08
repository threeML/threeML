---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Bayesian Posterior Sampling

When using Bayesian posterior sampling, we want to the posterior, ${\rm P}(M(\vec{\theta}) \vert D)$, of the model ($M(\vec{\theta})$)  given one or more datasets ($D$) (i.e., plugin instances) and one model containing one or more sources with free parameters $\vec{\theta} $ given a likelihood $L(\vec{\theta}) = {\rm P}(D \vert M(\vec{\theta}))$ and prior ${\rm P}(\theta)$ on the parameters. Only the simplest posteriors allow for an analytic solution, thus we must rely on Monte Carlo or nested sampling methods to sample the posterior. 

In 3ML, we currently provide two popular posterior sampling methods: [emcee](https://github.com/dfm/emcee) and [MULTINEST](https://github.com/farhanferoz/MultiNest). If you installed via conda, both packages are available, otherwise, only emcee is included.

## emcee

"[emcee](https://github.com/dfm/emcee) is an extensible, pure-Python implementation of Goodman & Weare's Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler." It uses multiple "walkers" to explore the parameter space of the posterior. For a complete understanding of the capabilites and limitations, we recommend a thorough reading of [Goodman & Weare (2010)](http://msp.org/camcos/2010/5-1/p04.xhtml). Nevertheless we emphasize these N points to keep in mind:

* emcee is for unimodal parameter estimation
* for complex likelihoods and marginal likelihood integration, check out thermodynamic integration (documentation coming soon)



Let's take a look at its usage for a simple likelihood.

<!-- #endregion -->

```python
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.seterr(all="ignore")
```


```python
%%capture
from threeML import *
from threeML.bayesian.tutorial_material import *
```

```python
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
silence_warnings()
set_threeML_style()


```

Let’s get a BayesianAnalysis object like the one we would have in a normal 3ML analysis. We use a custom function, prepared for this tutorial, which gives a BayesianAnalysis object having a very simple model with one free parameter ($\mu$), and with a likelihood having a very simple shape:

```python
# This returns a BayesianAnalysis object with a simple likelihood function,
# and the corresponding Model instance. These objects are what you will have
# in a typical 3ML analysis. The Model contains one point source, named "test",
# with a spectrum called "simple"
bayes, model = get_bayesian_analysis_object_simple_likelihood()
bayes.set_sampler('emcee')
# Let's look at the likelihood function, which in this illustrative example
# has a very simple shape
_ = plot_likelihood_function(bayes)
```

We must define a prior fo $\mu$ in order to sample. Let's use a uniform prior from 0 to 100. There are two ways to define this in 3ML:

```python
# This directly assigns the prior
model.test.spectrum.main.shape.mu.prior = Uniform_prior(lower_bound=1, upper_bound=100.)

# Alternatively, we can set an uniformative prior on the parameter's bounds
model.test.spectrum.main.shape.mu.set_uninformative_prior(Uniform_prior)

```

Ok, we are ready to sample. We will use only 5 walkers for this problem. We need to define a burn_in sample length which is how long we want to run the sampler to approach the target distribution so that we do not include these "learning" samples in our final results. Additionally, we need to declare the number of samples each walker will take.

```python
n_walkers = 5
burn_in = 10
n_samples = 500

bayes.sampler.setup(n_iterations=n_samples,n_burn_in=burn_in,n_walkers=n_walkers )

res = bayes.sample()
```

```python
_ = plot_sample_path(bayes,truth=40.,burn_in=n_walkers*burn_in)
```

We can see that we did not burn in the sampler long enough and we will have part of the burn in included in the results as can be seen in the marginal distribution of $\mu$:

```python
_ = bayes.results.corner_plot()
```

We could easily run the sampler longer, but let's reset the value of $\mu$ far from the target distribution and try more burn in samples.

```python
model.test.spectrum.main.shape.mu =99


n_walkers = 5
burn_in = 200
n_samples = 500

res = bayes.sample()
```

```python
_ = plot_sample_path(bayes,truth=40.,burn_in=n_walkers*burn_in)
_ = bayes.results.corner_plot()
```

The marginal distribution of the parameter $\mu$ *is* the answer for our parameter. It is the integral of the posterior over all other parameters, $\vec{\phi}$, given the data.

$${\rm P}(\mu \vert D) = \int_{-\infty}^{\infty} {\rm d}\vec{\phi} \; {\rm P}( \mu, \vec{\phi}\vert D )$$

Of course, in our simple model, there are no other parameters. The marginal natuarally includes the dependence of all other parameters and is only equivalent to MLE error estimates under assumptions of Gaussianity and linearity. 

We can sometimes use point-descriptors for the marginal, e.g., mean, median, etc., but when the marginal distribution is not easily symmetric, it can be more descriptive to plot it or speak of its highest posterior density interval:


```python
this_mu = bayes.results.get_variates('test.spectrum.main.Simple.mu')

print(this_mu.highest_posterior_density_interval(cl=.68))
print(this_mu.highest_posterior_density_interval(cl=.95))
```

### Priors

We have not paid much attention to our prior choice. So far, we have used an uninformative prior (except that we know in our example the range over which the parameter can be found), but what happens if we limit our prior such that we would never find the target distribution?

Let's set a uniform prior such that $\mu \in \{80-100 \}$.

```python
model.test.spectrum.main.shape.mu =99
model.test.spectrum.main.shape.mu.prior = Uniform_prior(lower_bound=80, upper_bound=100)

n_walkers = 2
burn_in = 200
n_samples = 1000

bayes.sampler.setup(n_iterations=n_samples,n_burn_in=burn_in,n_walkers=n_walkers )
res = bayes.sample()
```

```python
_ = plot_sample_path(bayes,truth=40.,burn_in=n_walkers*burn_in)
_ = bayes.results.corner_plot()
```

We see that the over this range, the marginal distribution is the same as the prior, i.e., uniformative. The "data" have not updated our information about $\mu$ and thus we cannot conclude anything. This is by design in our example, but if we were to believe that $\mu$ had to be found in the interval $\mu \in\{80-100\}$, the data have not helped us to determine anything.

Let's now look at setting a more informative prior on $\mu$. Suppose from either other measurements or physical insight, we are sure $\mu\sim 30$ with some uncertainty. Then we can impose a Gaussian prior over $\mu$.  

```python
model.test.spectrum.main.shape.mu =99
model.test.spectrum.main.shape.mu.bounds = (None, None)
model.test.spectrum.main.shape.mu.prior = Gaussian(mu=25,sigma=5)
```

```python
n_walkers = 2
burn_in = 200
n_samples = 1000

bayes.sampler.setup(n_iterations=n_samples,n_burn_in=burn_in,n_walkers=n_walkers )
res = bayes.sample()
```

Our prior information has biased the results to slightly lower values of $\mu$. Again, this is only to illustrate how to use different priors and their effects on the recovered results.

```python
_ = plot_sample_path(bayes,truth=40.,burn_in=n_walkers*burn_in)
_ = bayes.results.corner_plot()
```

## Complex Likelihood

We now examine a more complex likelihood.

```python
bayes, model = get_bayesian_analysis_object_complex_likelihood()

```

## MULTINEST

[MULTINEST](https://github.com/farhanferoz/MultiNest) is a nested sampling algorithm that is designed to integrate the the posterior to obtain the marginal likelihood. For details on the algorithm see [Feroz et al. (2009)](https://academic.oup.com/mnras/article-abstract/398/4/1601/981502), [Feroz et al. (2013)](https://arxiv.org/abs/1306.2144), and for details on the input arguments for the python interface we implement, see the documentation of [pymultinest](https://johannesbuchner.github.io/PyMultiNest/). If you find these algorithms useful for your research, please cite the originals authors!


Let's sample the complex likelihood from above with MULTINEST using 
<!-- #endregion -->

```python

bayes.set_sampler('multinest')


model.test.spectrum.main.shape.mu.prior = Uniform_prior(lower_bound=1, upper_bound=100)

_ = plot_likelihood_function(bayes)


bayes.sampler.setup(n_live_points=400, resume=False)


res = bayes.sample()
```

```python
_ = plot_sample_path(bayes)
_ = bayes.results.corner_plot()
```

MULTINEST has fully sampled the likelihood and broken apart the modes. Cleary point-descriptors of the results will be inefficent, but we use the output files of MULTINEST to do multi-modal analysis. For details, consult the MULTINEST documentation.

## UltraNest

Finally we can try with [ultranest](https://johannesbuchner.github.io/UltraNest/) which is another nested sampling algorithm.


```python
bayes.set_sampler('ultranest')
bayes.sampler.setup()


res = bayes.sample(quiet=True)
```

```python
_ = plot_sample_path(bayes)
_ = bayes.results.corner_plot()
```

## Accessing samples and error propagation

Error propagation with posterior samples is straight forward. Say we have a function $f(\mu) = Sin(\mu)$. For the complex likelihood, we would be out of luck if we were using MLE. However, we can directly calculate $f(\mu)$ with the samples from the posterior. 

```python
def f(mu):
    
    return mu**2
    
this_mu =  bayes.results.get_variates('test.spectrum.main.Complex.mu')
    
f_mu = f(this_mu)

print (f_mu)

fig, ax = plt.subplots()

ax.hist(f_mu.samples,bins=100)

_ =ax.set_xlabel(r'f($\mu$)')
```

As with any approach, the Bayesian capabilities of 3ML are a tool. Understanding and proper use of the posterior and marginal distributions requires special care. For further reading on the Bayesian analysis and its applications to spectral analysis check out the following resources:

* [Michael Betancourt](https://betanalpha.github.io/writing/) spends a ton of time trying to teach proper statistical methodology to applied fields. If you are using Bayesian techniques, you must read his work.

* [Frequentism and bayesianism: a python-driven primer](https://arxiv.org/abs/1411.5018)

* [Bayes in the sky: Bayesian inference and model selection in cosmology](http://www.tandfonline.com/doi/abs/10.1080/00107510802066753)

* [X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue](https://www.aanda.org/articles/aa/full_html/2014/04/aa22971-13/aa22971-13.html)

* [Analysis of energy spectra with low photon counts via Bayesian posterior simulation](http://iopscience.iop.org/article/10.1086/318656/meta)

```python

```
