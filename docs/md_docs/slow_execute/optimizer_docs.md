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
# Minimizer Examples

Examples of running each minimizer avaiable in 3ML.


Before, that, let's discuss setting up configuration default local minimizer. 
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
threeML_config.mle.default_minimizer
```

If you simply run `mle_analysis.fit()` the default minimizer and its default parameters will be used. 

```python
sin = Sin(K=.5, f=.5)
sin.phi.fix = True


# fixed seed, so that the simulated data are always the same
np.random.seed(0)

x = np.linspace(-1 * np.pi, 1 * np.pi, 50)
yerr = np.random.uniform(0.01, 0.05, 50)


xyl = XYLike.from_function("demo", sin, x, yerr )

xyl.plot();

sin = Sin(K=1, f=.5)
sin.phi.fix = True
sin.K.bounds = (0, 10)
sin.f.bounds = (0, 5)

model = Model(PointSource("demo",0,0,spectral_shape=sin))

dl = DataList(xyl)

mle_analysis = JointLikelihood(clone_model(model), dl)
```

## minuit

```python
local_minimizer = LocalMinimization("minuit")
local_minimizer.setup(ftol=1e-4)

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();

```

## ROOT

A minimizer based on the Minuit2 implementation of [ROOT](https://root.cern) is also available as ``LocalMinimization("ROOT")``, but only if PyROOT is installed (it is not part of the standard 3ML installation, and it is not used in this documentation).

<!-- #region heading_collapsed=true -->
## scipy
<!-- #endregion -->

```python hidden=true
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="L-BFGS-B")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();

```

```python hidden=true
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="TNC")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();


```

```python hidden=true
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="SLSQP")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();


```

## least_squares

A Levenberg–Marquardt-like minimizer, similar to the default fitting method of XSPEC. It minimizes the sum of the squares of the deviance residuals of each data point, which is equivalent to minimizing $-\log{L}$, using derivatives with steps proportional to the parameter values. It is robust for models with a limited numerical precision, such as many XSPEC models (see the Minimization tutorial for details).

It requires all the plugins to provide deviance residuals, which is the case for ``XYLike`` and for the spectral plugins (``SpectrumLike``, ``DispersionSpectrumLike``, ``OGIPLike``).

```python
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

local_minimizer = LocalMinimization("least_squares")

# diff_step: relative step of the derivatives (like "xset delta" in XSPEC)
# ftol: tolerance on the relative change of the statistic
local_minimizer.setup(diff_step=0.01, ftol=1e-8)

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();
```

The algorithm of scipy.optimize.least_squares can be selected between "trf" (default) and "dogbox":

```python
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

local_minimizer = LocalMinimization("least_squares")

local_minimizer.setup(algorithm="dogbox")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();
```

<!-- #region heading_collapsed=true -->
## GRID Minimizer
<!-- #endregion -->

```python hidden=true
# start every fit from the same point
mle_analysis.likelihood_model.demo.spectrum.main.Sin.K = 1
mle_analysis.likelihood_model.demo.spectrum.main.Sin.f = 0.5

grid_minimizer = GlobalMinimization("grid")

local_minimizer = LocalMinimization("minuit")


my_grid = {'demo.spectrum.main.Sin.K': np.logspace(-1, np.log10(5), 10)}

grid_minimizer.setup(
    second_minimization=local_minimizer, grid=my_grid)


mle_analysis.set_minimizer(grid_minimizer)

mle_analysis.fit();

xyl.plot();


```
