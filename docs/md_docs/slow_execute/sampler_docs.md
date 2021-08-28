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
# Bayesian Sampler Examples

Examples of running each sampler avaiable in 3ML.


Before, that, let's discuss setting up configuration default sampler with default parameters. We can set in our configuration a default algorithm and default setup parameters for the samplers. This can ease fitting when we are doing exploratory data analysis.

With any of the samplers, you can pass keywords to access their setups. Read each pacakges documentation for more details.

<!-- #endregion -->

```python
from threeML import *

import numpy as np

from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
silence_warnings()
set_threeML_style()
```

```python
threeML_config.bayesian.default_sampler
```

```python
threeML_config.bayesian.emcee_setup
```

<!-- #region -->
If you simply run `bayes_analysis.sample()` the default sampler and its default parameters will be used. 


Let's make some data to fit.
<!-- #endregion -->

```python
sin = Sin(K=1, f=.1)
sin.phi.fix = True
sin.K.prior = Log_uniform_prior(lower_bound=0.5, upper_bound=1.5)
sin.f.prior = Uniform_prior(lower_bound=0, upper_bound=.5)

model = Model(PointSource("demo",0,0,spectral_shape=sin))

x = np.linspace(-2 * np.pi, 4 * np.pi, 20)
yerr = np.random.uniform(.01,0.2, 20)


xyl = XYLike.from_function("demo",sin,x, yerr )
xyl.plot();

bayes_analysis = BayesianAnalysis(model, DataList(xyl))
```

## emcee

```python
bayes_analysis.set_sampler('emcee')
bayes_analysis.sampler.setup(n_walkers=20, n_iterations=500)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();

```

## multinest

```python
bayes_analysis.set_sampler('multinest')
bayes_analysis.sampler.setup(n_live_points=400, resume=False, auto_clean=True)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();

```

## dynesty

```python
bayes_analysis.set_sampler('dynesty_nested')
bayes_analysis.sampler.setup(n_live_points=400)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();


```

```python
bayes_analysis.set_sampler('dynesty_dynamic')
bayes_analysis.sampler.setup(n_live_points=400)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();


```

## zeus

```python
bayes_analysis.set_sampler('zeus')
bayes_analysis.sampler.setup(n_walkers=20, n_iterations=500)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();


```

## ultranest

```python
bayes_analysis.set_sampler('ultranest')
bayes_analysis.sampler.setup(min_num_live_points=400, frac_remain=0.5, use_mlfriends=False)
bayes_analysis.sample()

xyl.plot();
bayes_analysis.results.corner_plot();


```
