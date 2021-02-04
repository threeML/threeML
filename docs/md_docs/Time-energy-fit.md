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

# Time-energy fit

3ML allows the possibility to model a time-varying source by explicitly fitting the time-dependent part of the model. Let's see this with an example.

First we import what we need:

```python
import matplotlib.pyplot as plt
import numpy as np

from threeML import *
from threeML.io.package_data import get_path_of_data_file

```

```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
set_threeML_style()
silence_warnings()

import warnings
warnings.filterwarnings('ignore')
```



## Generating the datasets

Then we generate a simulated dataset for a source with a cutoff powerlaw spectrum with a constant photon index and cutoff but with a normalization that changes with time following a powerlaw:

```python
def generate_one(K, ax):

    # Let's generate some data with y = Powerlaw(x)

    gen_function = Cutoff_powerlaw()
    gen_function.K = K

    # Generate a dataset using the power law, and a
    # constant 30% error

    x = np.logspace(0, 2, 50)

    xyl_generator = XYLike.from_function(
        "sim_data", function=gen_function, x=x, yerr=0.3 * gen_function(x)
    )

    y = xyl_generator.y
    y_err = xyl_generator.yerr

    ax.loglog(x, gen_function(x))

    return x, y, y_err
```

These are the times at which the simulated spectra have been observed

```python
time_tags = np.array([1.0, 2.0, 5.0, 10.0])
```

This describes the time-varying normalization. If everything works as it should, we should recover from the fit a normalization of 0.23 and a index of -1.2 for the time law.

```python
normalizations = 0.23 * time_tags ** (-3.5)
```

Now that we have a simple function to create the datasets, let's build them.

```python tags=["nbsphinx-thumbnail"]
fig, ax = plt.subplots()

datasets = [generate_one(k, ax) for k in normalizations]

ax.set_xlabel("Energy")
ax.set_ylabel("Flux")
```

## Setup the model

Now set up the fit and fit it. First we need to tell 3ML that we are going to fit using an independent variable (time in this case). We init it to 1.0 and set the unit to seconds.

```python
time = IndependentVariable("time", 1.0, u.s)
```

Then we load the data that we have generated, tagging them with their time of observation.

```python

plugins = []

for i, dataset in enumerate(datasets):
    
    x, y, y_err = dataset
    
    xyl = XYLike("data%i" % i, x, y, y_err)
    
    # This is the important part: we need to tag the instance of the
    # plugin so that 3ML will know that this instance corresponds to the
    # given tag (a time coordinate in this case). If instead of giving
    # one time coordinate we give two time coordinates, then 3ML will
    # take the average of the model between the two time coordinates
    # (computed as the integral of the model between t1 and t2 divided 
    # by t2-t1)
    
    xyl.tag = (time, time_tags[i])
    
    # To access the tag we have just set we can use:
    
    independent_variable, start, end = xyl.tag
    
    # NOTE: xyl.tag will return 3 things: the independent variable, the start and the
    # end. If like in this case you do not specify an end when assigning the tag, end
    # will be None
    
    plugins.append(xyl)
```

Generate the datalist as usual



```python
data = DataList(*plugins)
```

Now let's generate the spectral model, in this case a point source with a cutoff powerlaw spectrum.

```python
spectrum = Cutoff_powerlaw()

src = PointSource("test", ra=0.0, dec=0.0, spectral_shape=spectrum)

model = Model(src)
```

Now we need to tell 3ML that we are going to use the time coordinate to specify a time dependence for some of the parameters of the model.



```python
model.add_independent_variable(time)
```

Now let's specify the time-dependence (a powerlaw) for the normalization of the powerlaw spectrum.

```python
time_po = Powerlaw()
time_po.K.bounds = (0.01, 1000)
```

Link the normalization of the cutoff powerlaw spectrum with time through the time law we have just generated.

```python
model.link(spectrum.K, time, time_po)
model
```

## Performing the fit

```python
jl = JointLikelihood(model, data)

best_fit_parameters, likelihood_values = jl.fit()
```

```python
for p in plugins:

    p.plot(x_scale='log', y_scale='log');
```

```python

```
