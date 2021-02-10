---
title: '3ML: The Multi-Mission Maximum Likelihood Framework'
tags:
  - Python
  - astronomy
  - inference
  - fitting
authors:
  - name: J Michael Burgess^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-3345-9515
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Max-Planck-Institut fur extraterrestrische Physik, Giessenbachstrasse 1, D-85748 Garching, Germany
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: paper.bib

<!-- # Optional fields if submitting to a AAS journal too, see this blog post: -->
<!-- # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing -->
<!-- aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it. -->
<!-- aas-journal: Astrophysical Journal <- The name of the AAS journal. -->
---

# Summary

Astrophysical sources are observed by different instruments at
different wavelengths with an unprecedented quality. Putting all these
data together to form a coherent view, however, is a very difficult
task. Indeed, each instrument and data type has its own ad-hoc
software and handling procedure, which present steep learning curves
and do not talk to each other.

The Multi-Mission Maximum Likelihood framework (3ML) provides a common
high-level interface and model definition, which allows for an easy,
coherent and intuitive modeling of sources using all the available
data, no matter their origin. At the same time, thanks to its
architecture based on plug-ins, 3ML uses under the hood the official
software of each instrument, the only one certified and maintained by
the collaboration which built the instrument itself. This guarantees
that 3ML is always using the best possible methodology to deal with
the data of each instrument.

Though Maximum Likelihood is in the name for historical reasons, 3ML
is an interface to several Bayesian inference algorithms such as MCMC
and nested sampling as well as likelihood optimization
algorithms. Each approach to analysis can be seamlessly switched
between allowing users to try different approaches quickly and without
having to rewrite their model or data interfaces.

# Background of Multi-wavelength Modeling Software

Multi-wavelength or spectral modeling software has existed for many
decades to handle the task of fitting astrophysical models to
observatory data. Most of these tools such as `XSPEC` `[@xspec]`, `Ciao` [@ciao] or
`Sherpa` [@sherpa] which primarily focus on the modeling of data at X-ray
wavelengths interface with data by enforcing a common data format such
as FITS [@fits] which implies that any observatory wishing to uses these
softwares to fit their data must for their data into this format. If a
user desires to fit data from multiple instruments, spanning several
wavelengths simultaneously, than all of this data must be correctly
formatted. However, many modern observatories (e.g. Fermi-LAT,
HAWC. etc) have data whose richness and complexity does not allow for
reformatting into the formats required for existing tools. Moreover,
most of these modern observatories have complex analysis algorithms
expressing detailed likelihoods which can not interface with existing
multi-wavelength software; much less with each other.

# Statement of need

`3ML` provides an abstract data interface via a plugin architecture to
various astrophysical observatories. As instruments grow more complex
in their observational processes, it is important that their data are
properly analyzed with the software and likelihood that is appropriate
for that data. By creating plugins that interface a high-level
astrophysical modeling language to an instrument's specific data
likelihood, 3ML allows for data from multiple instruments to be fit
together while maintaining each instruments maximum sensitivity and
resolution.

Many tools exists that provide some of the functionality of 3ML such
as `XSPEC`, `Sherpa`, and `XXX`. However, are primarily designed for
spectral fitting of x-ray data and require several data reduction
steps of non x-ray data for that data to be loaded into these programs
and jointly fitted with other instruments. This has the effect of
introducing systematics into the fitting process as well as
potentially reducing the sensitivity of these instruments. Moreover,
3ML allows for the simultaneous fitting of model spaces other than
purely spectral such as polarization, spatial, temporal, etc. Finally,
3ML is built on a modern, modular, python object-oriented framework
natives rather than wrapping legacy software as is the approach of
some other existing tools. This wrapping results in a rigid singleton
that is prohibitive to the construction of data pipelines and parallel
computation in HPC environments.

# Software Description
The core functionality of `3ML` is based upon a plugin system. Plugins
for various instruments and data types are inherited from a base
`PluginPrototype` class. The inherited plugin must provide two main
functions: 1) a function that translates a model into the instrument's
specific software and 2) a function that returns the logarithm of the
likelihood between the instruments data and a given parameterization
of the translated model. The passed model which is from our
accompanying software, `astromodels`, is shared among all plugins in
an analysis. Thus as the parametrization of the model changes during
optimization or sampling, the likelihood in each plugin is
modified. When a set of plugins is passed to one of 3ML's maximum
likelihood or Bayesian analysis classes, they talk to each other via
the summing of their logarithmic likelihoods.
	
With this framework, plugins can essentially wrap existing low-level
instrument analysis software and allow this software to communicate to
other instrument software via the likelihood in an analysis. The
resolution and sensitivity of instruments are maintained, and there is
no need to format data into a file type or format that reduces the
richness of the native instrument data. 

3ML also extends beyond multi-wavelength data to the multi-messenger
regime by providing abstract interfaces for modeling polarization,
particle and spatial data. MORE


# Data Analysis
Several optimization (`iminuit`[], `scipy`[], `pagmo`[]) and sampling
(`emcee`[], `multinest`[], `zeus`[], `dynesty`[], `ultranest`[]) are
provided allowing for the user to flexibly fit multi-messenger data in
a workflow that suits their needs. Switching between various
algorithms is streamline by providing a nearly identical interface to
both maximum likelihood and Bayesian analysis
procedures. Additionally, all fir results are fully serializible to
disk allowing for the distribution of the full richness of an analysis
and thus enabling easy replication of results.


# History
`3ML` began as an effort to be able to properly fit the data of
Fermi-LAT and GBM as well as Fermi-LAT and HAWC together in a
statistically robust way that preserved the sensitivity of all
instruments involved in the analysis. No existing software allowed for
this. Once the concept of plugins was realized, it was easy to
generalize the process to be extended to any instrument and data type
that could follow the plugin architecture. The plugins currently
included as part of the core software extend from optical to very
high-energy wavelengths as well as polarization data.



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Figures

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->
<!-- and referenced from text using \autoref{fig:example}. -->

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
