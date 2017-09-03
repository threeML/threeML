.. The Multi-Mission Maximum Likelihood framework documentation master file, created by
   sphinx-quickstart on Fri Feb  5 12:26:57 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../logo/logo.png

Astrophysical sources are observed by different instruments at different wavelengths with an unprecedented quality. Putting all these data together to form a coherent view, however, is a very difficult task. Indeed, each instrument and data type has its own ad-hoc software and handling procedure, which present steep learning curves and do not talk to each other.

The Multi-Mission Maximum Likelihood framework (3ML) provides a common high-level interface and model definition which allows for an easy, coherent and intuitive modeling of sources using all the available data, no matter their origin. At the same time, thanks to its architecture based on plug-ins, 3ML uses under the hood the official software of each instrument, the only one certified and maintained by the collaboration which built the instrument itself. This guarantees that 3ML is always using the best possible methodology to deal with the data of each instrument.

Traditionally the Astrophysics community have been using frequentist techniques, but in recent years Bayesian methods and approaches have been gaining consensum and momentum. In 3ML both analysis are possible. Moreover, the 3ML Python interface allows for combinations with all available packages for data analysis and mining.

.. toctree::
    :maxdepth: 5
    :hidden:

    intro
    notebooks/Minimization_tutorial.ipynb
    notebooks/Bayesian_tutorial.ipynb
    notebooks/Time-energy-fit.ipynb
    plugins
    features
    

.. automodule:: threeML
    :members:
