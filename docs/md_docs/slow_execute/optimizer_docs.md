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

```python
local_minimizer = LocalMinimization("ROOT")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();
```

## scipy

```python
local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="L-BFGS-B")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();

```

```python
local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="TNC")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();


```

```python
local_minimizer = LocalMinimization("scipy")

local_minimizer.setup(algorithm="SLSQP")

mle_analysis.set_minimizer(local_minimizer)

mle_analysis.fit();

xyl.plot();


```

## GRID Minimizer

```python
grid_minimizer = GlobalMinimization("grid")

local_minimizer = LocalMinimization("minuit")


my_grid = {'demo.spectrum.main.Sin.K': np.logspace(-1, np.log10(5), 10)}

grid_minimizer.setup(
    second_minimization=local_minimizer, grid=my_grid)


mle_analysis.set_minimizer(grid_minimizer)

mle_analysis.fit();

xyl.plot();


```
