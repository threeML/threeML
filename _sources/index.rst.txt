.. The Multi-Mission Maximum Likelihood framework documentation master file, created by
   sphinx-quickstart on Fri Feb  5 12:26:57 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

3ML
===
  
.. image:: ../logo/logo.png

Astrophysical sources are observed by different instruments at different wavelengths with an unprecedented quality. Putting all these data together to form a coherent view, however, is a very difficult task. Indeed, each instrument and data type has its own ad-hoc software and handling procedure, which present steep learning curves and do not talk to each other.

The Multi-Mission Maximum Likelihood framework (3ML) provides a common high-level interface and model definition, which allows for an easy, coherent and intuitive modeling of sources using all the available data, no matter their origin. At the same time, thanks to its architecture based on plug-ins, 3ML uses under the hood the official software of each instrument, the only one certified and maintained by the collaboration which built the instrument itself. This guarantees that 3ML is always using the best possible methodology to deal with the data of each instrument.

.. image:: /media/plugin_demo.png

Though **Maximum Likelihood** is in the name for historical reasons, 3ML is an interface to several **Bayesian** inference algorithms such as MCMC and nested sampling as well as likelihood optimization algorithms. Each approach to analysis can be seamlessly switched between allowing users to try different approaches quickly and without having to rewrite their model or data interfaces. 


.. toctree::
    :maxdepth: 5
    :hidden:
    
    notebooks/installation.ipynb
    intro
    notebooks/configuration.ipynb
    notebooks/logging.ipynb
    xspec_users
    notebooks/Minimization_tutorial.ipynb
    notebooks/Bayesian_tutorial.ipynb
    notebooks/sampler_docs.ipynb
    plugins
    modeling
    faq
    api/API
    release_notes

.. nbgallery::
   :caption: Features and examples:

   notebooks/Analysis_results_showcase.ipynb
   notebooks/random_variates.ipynb
   notebooks/Point_source_plotting.ipynb
   notebooks/Building_Plugins_from_TimeSeries.ipynb
   notebooks/grb080916C.ipynb
   notebooks/APEC_doc.ipynb
   notebooks/joint_BAT_gbm_demo.ipynb
   notebooks/joint_fitting_xrt_and_gbm_xspec_models.ipynb
   notebooks/flux_examples.ipynb
   notebooks/Fermipy_LAT.ipynb
   notebooks/LAT_Transient_Builder_Example.ipynb
   notebooks/Time-energy-fit.ipynb
   notebooks/synthetic_spectra.ipynb
   notebooks/gof_lrt.ipynb
   
    
ThreeML is supported by the National Science Foundation (NSF) 

.. image:: ../logo/NSF_4-Color_bitmap_Logo.png
