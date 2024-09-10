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

## What happens to the model/function during a fit?

The spectral/spatial shapes that are input into the models and subsequently used during the fit are objects. There parameters are members of those objects and when they are changed by the user or the fitting engine, the parameter values in those objects are modified. 

```python
%%capture
from threeML import *

power_law = Powerlaw()

print("power law index before change:")
print(power_law.index)

power_law.index = 0

print("power law index after change:")
print(power_law.index)


# or create a power law with a different default index
power_law = Powerlaw(index=-1.5)

print("power law index after creation:")
print(power_law.index)


```

```python
import numpy as np
x = np.logspace(0, 2, 50)

xyl_generator = XYLike.from_function("sim_data", function = power_law, 
                                     x = x, 
                                     yerr = 0.1 * power_law(x))

y = xyl_generator.y
y_err = xyl_generator.yerr

fit_function = Powerlaw()

print("power law index before fit:")
print(fit_function.index)

xyl = XYLike("data", x, y, y_err)

results = xyl.fit(fit_function)


print("power law index after fit:")
print(fit_function.index)
```

After a fit, the fitted result are stored in an AnalysisResults object so that if the fit function's values are further modified, the best fit parameters can still be recovered.


## Why does my plugin not return a get_log_like()?

When a plugin is created, it does not have a likelihood model set initially. This is typically done when a DataList containing the plugin is passed to a JointLikelihood or BayesianAnalysis constructor along with a model. One can manually pass a model object to the plugin using the set_model() member of the plugin. 


## Why did my plugin lose its model?

If you use the same plugin with different models bvy passing it to successive JointLikelihood or BayesianAnalysis constructors, the plugin will have the last model with which it was used set as its model. 
