Release Notes
=============


Version 1.2
-----------


v1.2.0
^^^^^^^^
*Fri, 11 Sep 2020 23:42:47 + 0000*

* Added functionality to `FermiLatLike` plugin. Merged in pull request:
  https://github.com/threeML/threeML/pull/368
* Fix for results keys in `ultranest` sampler. Merged in pull request:
  https://github.com/threeML/threeML/pull/367
* Fix for `int` to `float` conversion issue of `numba` array. Merged in pull
  request: https://github.com/threeML/threeML/pull/373
* Fix to LAT data downloader in case of multiple files. Merged in pull 
  request: https://github.com/threeML/threeML/pull/376
* Added `numdifftools` to requirements and fixed a test. Merged in pull
  request: https://github.com/threeML/threeML/pull/375
* Issue(s) closed:

  * https://github.com/threeML/threeML/issues/356
  * https://github.com/threeML/threeML/issues/372


Version 1.1
-----------


v1.1.1
^^^^^^^^
*Mon, 11 May 2020 21:17:58 + 0000*

* Added `dynesty >= 1` version dependency.
* Fixed cmap, `multinest` import and install script bugs.


v1.1.0
^^^^^^^^
*Thu, 30 Apr 2020 04:19:52 + 0000*

* Added the ability to build BALROG. Merged in pull request:
  https://github.com/threeML/threeML/pull/362
* Fix for the `str`/`unicode` issue in python 2.


Version 1.0
-----------


v1.0.9
^^^^^^^^
*Tue, 28 Apr 2020 01:31:32 + 0000*

* Interface to `zeus` updated. Merged in pull request:
  https://github.com/threeML/threeML/pull/360
* Added `dynesty` sampler. Merged in pull request:
  https://github.com/threeML/threeML/pull/361


v1.0.8
^^^^^^^^
*Sat, 25 Apr 2020 02:27:06 + 0000*

* Added `numba` likelihoods. Merged in pull request:
  https://github.com/threeML/threeML/pull/359


v1.0.7
^^^^^^^^
*Wed, 22 Apr 2020 19:22:43 + 0000*

* Fixed some bugs in plotting and reading plugins with a background model. 
  Merged in pull request: https://github.com/threeML/threeML/pull/358


v1.0.6
^^^^^^^^
*Fri, 17 Apr 2020 18:27:31 + 0000*

* Fixed a bug in the `ResidualPlot` of spectra.


v1.0.5
^^^^^^^^
*Fri, 17 Apr 2020 06:57:47 + 0000*

* Documentation updated with new gallery for example. Merged in pull request:
  https://github.com/threeML/threeML/pull/351
* Issue(s) closed:

  * https://github.com/threeML/threeML/issues/355


v1.0.4
^^^^^^^^
*Wed, 15 Apr 2020 07:58:00 + 0000*

* Readme and Python versions updated.


v1.0.3
^^^^^^^^
*Wed, 15 Apr 2020 01:37:00 + 0000*

* Removed `pygmo` from requirements of `pip install` causing a failure. Merged
  in pull request: https://github.com/threeML/threeML/pull/350
* Setting of model moved back in the `BayesianAnalysis` class. Merged in pull 
  request: https://github.com/threeML/threeML/pull/353
* Fixed a bug in background loading when file is an empty string. Merged in pull
  request: https://github.com/threeML/threeML/pull/354
* Issue(s) closed:

  * https://github.com/threeML/threeML/issues/352



v1.0.2
^^^^^^^^
*Sat, 11 Apr 2020 06:49:00 + 0000*

* New interface to the bayesian sampling and introdution of two new samplers 
  (`ultranest` and `zeus`). Merged in pull request: 
  https://github.com/threeML/threeML/pull/349


v1.0.1
^^^^^^^^
*Fri, 10 Apr 2020 07:57:00 + 0000*

* Solved an issue with travis causing a failure in the upload to PyPI.


v1.0.0
^^^^^^^^
*Fri, 10 Apr 2020 01:38:00 + 0000*

* Added Python 3 compatibility. Merged in pull request:
  https://github.com/threeML/threeML/pull/346
* New system to manage software versioning and upload to conda/pypi. Merged in
  pull request: https://github.com/threeML/threeML/pull/347