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

# Spectral Models

Spectral models are provided via astromodels. For details, visit the astromodels [documentation](http://astromodels.readthedocs.io/en/latest/Model_tutorial.html). 

The important points are breifly covered below. 




## Building Custom Models

One of the most powerful aspects of astromodels and 3ML is the ability to quickly build custom models on the fly. The source code for a model can be pure python, FORTRAN linked via f2py, C++ linked via cython, etc. Anything that provides a python function can be used to fit data. 

To build a custom spectral model in 3ML, we need to import a few things that will allow astromodels to recognize your model.

```python
from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation
```

<!-- #region -->
Function1D is the base class for 1D spectral models and FunctionMeta is ABC class that ensures all the needed parts of a model are in the class as well as making the class function as it should.


There are three basic parts to declaring a model:

* the docstring
* the units setter
* the evaluate function

Let's look at the simple case of the power law already define in astromodels.

<!-- #endregion -->

```python
class Powerlaw(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A simple power-law
        latex : $ K~\frac{x}{piv}^{index} $
        parameters :
            K :
                desc : Normalization (differential flux at the pivot value)
                initial value : 1.0
                is_normalization : True
                transformation : log10
                min : 1e-30
                max : 1e3
                delta : 0.1
            piv :
                desc : Pivot value
                initial value : 1
                fix : yes
            index :
                desc : Photon index
                initial value : -2
                min : -10
                max : 10
      
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.index.unit = astropy_units.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            self.piv.unit = x_unit

            # The normalization has the same units as the y

            self.K.unit = y_unit

        # noinspection PyPep8Naming
        def evaluate(self, x, K, piv, index):

            xx = np.divide(x, piv)

            return K * np.power(xx, index)


```

<!-- #region -->
### The docstring

We have used the docstring interface to provide a YAML description of the model. This sets up the important information used in the fitting process and record keeping. The docstring has three parts:

- description
    - The description is a text string that provides readable info about the model. Nothing fancy, but good descriptions help to inform the user.
- latex
    - If the model is analytic, a latex formula can be included
- parameters
    - For each parameter, a description and initial value must be included. Transformations for fitting, min/max values and fixing the parameter can also be described here.
    
    
Keep in mind that this is in YAML format.

### Set units

3ML and astromodels keep track of units for you. However, a model must be set up to properly describe the units with astropy's unit system. Keep in mind that models are fit with a differential photon flux, 

$$\frac{d N_p}{dA dt dE}$$

so your units should reflect this convention. Therefore, proper normalizations should be taken into account.


### Evaluate
This is where the function is evaluated. The first argumument **must be called x** and the parameter names and ordering must reflect what is in the docstring. Any number of operations can take place inside the evaluate call, but remember that the return must be in the form of a differential photon flux. 


A model is defined in a python session. If you save the results of a fit to an AnalysisResults file and try to load this file without loading this model, you will get a error,


## Custom models in other langauges

What if your model is built from a C++ function and you want to fit that directly to the data? using Cython, pybind, f2py, etc, you can wrap these models and call them easily.
<!-- #endregion -->

```python

def cpp_function_wrapper(a):
    # we could wrap a c++ function here
    # with cython, pybind11, etc
    
    return a

```

```python
cpp_function_wrapper(2.)
```

Now we will define a spectral model that will handle both the unit and non-unit call.

```python
import astropy.units as astropy_units

class CppModel(Function1D,metaclass=FunctionMeta):
        r"""
        description :
            A spectral model wrapping a cython function
        latex : $$
        parameters :
            a :
                desc : Normalization (differential flux)
                initial value : 1.0
                is_normalization : True
                min : 1e-30
                max : 1e3
                delta : 0.1
        """

        def _set_units(self, x_unit, y_unit):

            # The normalization has the same units as the y

            self.a.unit = y_unit

        
        def evaluate(self, x, a):
            
            # check is the function is being called with units
            
            if isinstance(a, astropy_units.Quantity):
                
                # get the values
                a_ = a.value
                
                # save the unit
                unit_ = self.y_unit
                
            else:
                
                # we do not need to do anything here
                a_ = a
                
                # this will basically be ignored
                unit_ = 1.

            # call the cython function
            flux = cpp_function_wrapper(a_)

            # add back the unit if needed
            return flux * unit_
```

We can check the unit and non-unit call by making a point source and evaluating it

```python
cpp_spectrum = CppModel()

from astromodels import PointSource

point_source = PointSource('ps',0,0,spectral_shape=cpp_spectrum)

print(point_source(10.))
point_source(10. * astropy_units.keV)
```

## Template (Table) Models

3ML (via astromodels) provides the ability to construct models in tabluated form. This is very useful for models that are from numerical simualtions. While in other software special care must be taken to construct table models into FITS files, in 3ML the construction of the table model is taken care of for you. Here is an example of how to build a template model from a pre-existing function.

### Constructing a template
First, we grab a function and make an energy grid.

```python
from astromodels import Band
import numpy as np

model = Band()

# we won't need to modify the normalization
model.K = 1.


# if no units are provided for the energy grid, keV will be assumed!
energies = np.logspace(1, 3, 50)
```

Now we define a template model factory. This takes a name, a description, the energy grid and an array of parameter names as input.

```python
from astromodels import TemplateModelFactory
tmf = TemplateModelFactory('my_template', 'A test template', energies, ['alpha', 'xp', 'beta'])
```

Now, define our grid in parameter space. While we are using a function here, this grid could be from a text file, a database of simulations, etc. We then assign these grid points to the template model factory.

```python
alpha_grid = np.linspace(-1.5, 1, 15)
beta_grid = np.linspace(-3.5, -1.6, 15)
xp_grid = np.logspace(1, 3, 20)



tmf.define_parameter_grid('alpha', alpha_grid)
tmf.define_parameter_grid('beta', beta_grid)
tmf.define_parameter_grid('xp', xp_grid)
```

Finally, we loop over our grid and set the interpolation data to the template model factory. The units of the fluxes must be a differential photon flux! 

```python
for a in alpha_grid:

    for b in beta_grid:

        for xp in xp_grid:
            
            # change our model parameters
            model.alpha = a
            model.beta = b
            model.xp = xp

            tmf.add_interpolation_data(model(energies), alpha=a, xp=xp, beta=b)

```

We can now save our model to disk. The formal is an HDF5 file which is saved to the astromodel data directory (~/.astromodels/data). The HDF5 file can easily be passed around to other users as all information defining the model is stored in the file. The other user would place the file in thier astromodels data directory.

```python
tmf.save_data(overwrite=True)

from astromodels import TemplateModel

reloaded_table_model = TemplateModel('my_template')
```

```python
reloaded_table_model(energies)
```

```python

```
