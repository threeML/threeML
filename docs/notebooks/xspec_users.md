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

# Notes for XSPEC Users


Users coming from XSPEC typically have a few questions about getting started with 3ML


## Do I need a new plugin for my instrument?
* If it is an X-ray instrument that has PHA1 data, BAK files, RSPs and ARFs, nope! This is handled by the OGIPLike plugin. 
* Think of OGIPLike to be a XSPEC-like object. Feed your data in and pass it to to the JointLikelihood or BayesianAnalysis objects. You need one plugin per observation. 
* OGIPLike is simply provides a wrapper around DispersionSpectrumLike that reads standard OGIP files. We are strict about following the OGIP standard.



## Can I use XSPEC models?
* Yes!
* astromodels provides and interface to XSPEC models. See details in the modeling section.
* We are currently building our own set of standard models in XSPEC. We already have APEC, PhAbs, Wabs, Tbabs etc. So you can try those out first. 


## How do I fake a dummy response to fit optical data or a background model?
* DON'T DO THAT!
* Since 3ML is not limited to a rigid data format, we have custom plugins for photometric data. You simply need to provide the filter name and magnitude. See the docs for more details. 
* We have the ability to model background spectra along with source spectrum. Check out the background modeling n the docs.

## How do I choose the likelihood statistic for my fit?
* It is possible, but if your PHA files are formatted correctly, we probe them and choose the **correct** likelihood for your data.
* We do support rebinning of data, but not for the purposes of moving from the Poisson regime to the so-called $\chi^2$ regime. This is an incorrect and bad practice. 

## Can I trust the results of the fits?
* With each build of 3ML, we always test the code automatically against XSPEC to ensure fitting (up to factor) and RSP convolution give the same results.
* You can always try yourself as the file types are the same. Expect differences in fit results that could be due to the underlying fitting engines.


```python

```
