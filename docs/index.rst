.. The Multi-Mission Maximum Likelihood framework documentation master file, created by
   sphinx-quickstart on Fri Feb  5 12:26:57 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../logo/logo.png

Astrophysical sources are observed by different instruments at different wavelengths with an unprecedented quality. Putting all these data together to form a coherent view, however, is a very difficult task. Indeed, each instrument and data type has its own ad-hoc software and handling procedure, which present steep learning curves and do not talk to each other.

The Multi-Mission Maximum Likelihood framework (3ML) provides a common high-level interface and model definition, which allows for an easy, coherent and intuitive modeling of sources using all the available data, no matter their origin. At the same time, thanks to its architecture based on plug-ins, 3ML uses under the hood the official software of each instrument, the only one certified and maintained by the collaboration which built the instrument itself. This guarantees that 3ML is always using the best possible methodology to deal with the data of each instrument.

.. image:: /media/plugin_demo.png

Though **Maximum Likelihood** is in the name for historical reasons, 3ML is an interface to several **Bayesian** inference algorithms such as MCMC and nested sampling as well as likelihood optimization algorithms. Each approach to analysis can be seamlessly switched between allowing users to try different approaches quickly and without having to rewrite their model or data interfaces. 



.. toctree::
    :maxdepth: 5
    :hidden:
    
    installation
    intro
    notebooks/configuration
    notebooks/logging
    notebooks/Minimization_tutorial
    notebooks/Bayesian_tutorial
    plugins
    modeling
    faq
    api/API
    release_notes

.. nbgallery::
   :caption: Features and examples:

   notebooks/grb080916C
   notebooks/joint_BAT_gbm_demo
   notebooks/Time-energy-fit
   notebooks/Analysis_results_showcase
   notebooks/random_variates
   notebooks/Point_source_plotting
   notebooks/synthetic_spectra
   notebooks/gof_lrt
    
