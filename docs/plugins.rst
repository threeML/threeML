Plugins
=======

3ML is based on a plugin system. This means that For each instrument/datum, there is a plugin that holds the data, reads a model, and returns a likelihood.
This is how we achieve the multi-messenger paradigm. A plugin handles its likelihood call internally and the likelhoods are combined within 3ML during a fit.

Contents:

.. toctree::
   :maxdepth: 4

   notebooks/custom_plugins.ipynb
   notebooks/spectrum_tutorial.ipynb
   notebooks/Building_Plugins_from_TimeSeries.ipynb
   notebooks/Background_modeling.ipynb
   notebooks/Photometry_demo.ipynb
