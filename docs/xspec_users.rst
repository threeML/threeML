Notes for XSPEC Users
=====================

Users coming from XSPEC typically have a few questions about getting started with 3ML

What version of XSPEC is currently supported?
---------------------------------------------

Starting from version ``astromodels`` and ``threeML`` 2.5.0 there is currently 
support for XSPEC, either installed with Conda via the `xspec` conda package or 
compiled from source. (see `here <https://heasarc.gsfc.nasa.gov/docs/software/conda.html>`_ for more details on the conda package).

`astromodels` 2.5.0 is compatible with XSPEC 12.15.0, while 2.5.1 enables 
support for XSPEC 12.15.1. XSPEC versions lower than 12.12.0 are no longer
supported. 

If you compiled XSPEC from source, please set the environment variable
ASTRO_XSPEC_VERSION to the version of XSPEC you are using in the build
process and make sure to have the HEADAS environment variable set.

Support for `xspec-modelsonly` conda package has been discontinued.
Please use the conda package `xspec` instead.

In `astromodels` 2.5.1 the internal logic for the normalization parameter 
for XSPEC additive models has changed to be consistent with `sherpa` and
support XSPEC 12.15.1. This means that XS model files in the user data
folder (usually `~/.astromodels/data/`) need to be regenerated. This should
be automatically taken care of and transparent to the users, but in case of 
errors, please remove these files manually and try to import XSPEC models 
again to trigger the new file generation.


Why use 3ML over XSPEC if I am just fitting X-ray data?
-------------------------------------------------------

XSPEC is an amazing tool built by dedicated, friendly experts. If XSPEC is your favorite tool and you are only fitting X-ray data, then there are some marginal benefits:

- 3ML is a lighter install and natively in Python rather than a Python wrapper around XSPEC
- 3ML does many operations in parallel on HPC environments with little effort
- 3ML has many many more optimizers and Bayesian samplers
   * `minuit <https://iminuit.readthedocs.io/en/stable/>`__
   * `PAGMO <https://esa.github.io/pagmo2/> (dozens and dozens of methods)`__
   * `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__
   * least squares (Levenberg-Marquardt-like, similar to ``method leven`` in XSPEC)
   *  global and local optimizer combinations
   * `emcee <https://emcee.readthedocs.io/en/stable/>`__
   * `multinest <https://github.com/farhanferoz/MultiNest>`__
   * `dynesty <https://dynesty.readthedocs.io/en/latest/>`__
   * `zeus <https://zeus-mcmc.readthedocs.io/en/latest/>`__
   * `ultranest <https://johannesbuchner.github.io/UltraNest/>`__
   * and more as they come into existence
- No parsing of text files to save results! 3ML has a serializable fit results format (FITS or HDF5) that is portable and machine readable.
- The modularity of 3ML means that the types of data it can fit will grow quickly


Do I need a new plugin for my instrument?
-----------------------------------------
- If it is an X-ray instrument that has PHA1 data, BAK files, RSPs and ARFs, nope! This is handled by the OGIPLike plugin. 
- Think of OGIPLike as an XSPEC-like object. Feed your data in and pass it to to the JointLikelihood or BayesianAnalysis objects. You need one plugin per observation. 
- OGIPLike is simply provides a wrapper around DispersionSpectrumLike that reads standard OGIP files. We are strict about following the OGIP standard.


Can I use XSPEC models?
-----------------------
- Yes!
- astromodels provides and interface to XSPEC models. See details in the modeling section.
- We are currently building our own set of standard models in XSPEC. We already have **APEC**, **PhAbs**, **Wabs**, **Tbabs** etc. So you can try those out first. 

Can I trust the results of the fits? 
------------------------------------
.. image:: https://github.com/threeML/threeML/workflows/Test%20Against%20XSPEC/badge.svg
- With each build of 3ML, we always test the code automatically against XSPEC to ensure fitting (up to factor) and RSP convolution give the same results.
- You can always try yourself as the file types are the same. Expect differences in fit results that could be due to the underlying fitting engines.


How do I fake a dummy response to fit optical data or a background model?
-------------------------------------------------------------------------
- DON'T DO THAT!
- Since 3ML is not limited to a rigid data format, we have custom plugins for photometric data. You simply need to provide the filter name and magnitude. See the docs for more details. 
- We have the ability to model background spectra along with source spectrum. Check out the background modeling n the docs.

How do I choose the likelihood statistic for my fit?
----------------------------------------------------
- It is possible, but if your PHA files are formatted correctly, we probe them and choose the **correct** likelihood for your data.
- We do support rebinning of data, but not for the purposes of moving from the Poisson regime to the so-called :math:`\chi^2` regime. This is an incorrect and bad practice.

Does 3ML use the GROUPING of my PHA file?
-----------------------------------------
- Not automatically: by default every channel is used separately.
- To group the channels as XSPEC does, call ``rebin_on_grouping()`` on the plugin after selecting the energy range. The grouping is read from the GROUPING column of the PHA file:

.. code-block:: python

    nustar = OGIPLike("FPMA", observation="src_grouped.pha", background="bkg.pha",
                      response="src.rmf", arf_file="src.arf")

    nustar.set_active_measurements("3-79")
    nustar.rebin_on_grouping()

- Channels outside of the selection are dropped, so a group crossing the edge of the selection keeps only its selected channels. To change the selection afterwards, call ``remove_rebinning()`` first.
- You can also pass your own grouping, as an array of OGIP grouping values (1 at the first channel of a group, -1 for the following channels): ``nustar.rebin_on_grouping(my_grouping)``.

My fit with an XSPEC model does not converge (or takes forever) with MINUIT
---------------------------------------------------------------------------
- Many XSPEC models are computed in single precision, or solve numerical equations internally (e.g. ``eqpair``). Their output, and therefore the likelihood, is not smooth at small scales. MINUIT computes derivatives with very small steps, so it can fail to converge, or wander to extreme parameter values where some models become very slow.
- Use the ``least_squares`` minimizer, which works like the default XSPEC fitting method (``method leven``): the curvature is built from first derivatives of the residuals in each bin, and the derivative steps are proportional to the parameter values (1% by default, like ``xset delta 0.01``):

.. code-block:: python

    jl = JointLikelihood(model, data)
    jl.set_minimizer("least_squares")
    jl.fit()

- The step and the tolerance can be changed with:

.. code-block:: python

    least_squares = LocalMinimization("least_squares")
    least_squares.setup(diff_step=0.01, ftol=1e-8)
    jl.set_minimizer(least_squares)

- The errors returned by the fit come from the Gauss-Newton approximation of the Hessian (as in XSPEC). Use ``jl.get_errors()`` and ``jl.get_contours()`` (the equivalents of ``error`` and ``steppar``) for profile likelihood errors and scans.
- As in XSPEC, keep the parameters away from special values at their boundaries, where some models switch to a different computation (for example ``xi = 0`` in ``eqpair``), and set reasonable bounds for parameters that can make the model very slow.
- ``least_squares`` requires all the plugins to provide residuals for each bin. This is the case for all the spectral plugins (``SpectrumLike``, ``DispersionSpectrumLike``, ``OGIPLike``), except when the background is modeled with another plugin, and for ``XYLike``.

How do I plot *unfolded data points* along with my fit?
-------------------------------------------------------
- You can't and you shouldn't
