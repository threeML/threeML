Installation
============

.. toctree::
   :maxdepth: 2

Suggested method
~~~~~~~~~~~~~~~~

Install 3ML with the automatic script which will take care of everything. The script uses `Conda`_ , which is a
platform independent package manager.

1. Download the script from `here`_
2. Run the script. If you plan to use XSpec models use
``bash install_3ML.sh --with-xspec``. If you want to use the HAWC plugin, the VERITAS plugin or other features
of 3ML needing ROOT, use ``--with-root``. Of course you can use both options at the same time:
``bash install_3ML.sh --with-root --with-xspec``. If you do not need either, you can use just
``bash install_3ML.sh``. The script will download Miniconda if needed (or use your existing conda installation),
create a new environment for 3ML, and install all the needed software in such environment. Thanks to this, the 3ML
installation will not change anything on your system and can be removed by removing the `threeML`
conda environment (``conda uninstall --name threeML --all``)
3. The script will install 3ML and then create a ``threeML_init.sh`` script and a ``threeML_init.csh`` script
in the directory where you launched it. These scripts can be moved anywhere. Source the former if you are using
Bash (``source threeML_init.sh``) and the latter if you are using Csh/Tcsh (``source threeML_init.csh``)

In order to use the HAWC plugin, you will also need to install cthreeML
(run this *after* setting up the HAWC environment):

.. code:: bash

    > source threeML_init.sh
    > [setup HAWC environment as usual]
    > export CFLAGS="-m64 -I${CONDA_PREFIX}/include"
    > export CXXFLAGS="-DBOOST_MATH_DISABLE_FLOAT128 -m64 -I${CONDA_PREFIX}/include"
    > pip install git+https://github.com/giacomov/cthreeML.git --no-deps --upgrade

Manual method
~~~~~~~~~~~~~

If you are familiar with Conda and you already have it installed, you
can install threeML on your own. Start by creating an environment (highly suggested) with:

.. code:: bash

    conda create --name threeML -c conda-forge python=2.7 numpy scipy matplotlib

Then activate your environment and install 3ML as:

.. code:: bash

    source activate threeML
    conda install -c conda-forge -c threeml threeml

If you need XSpec support and/or ROOT support, you need to install also the respective packages ``root5``
and ``xspec-modelsonly`` with:

.. code:: bash

    source activate threeML
    conda install -c conda-forge -c threeml [package]

Other dependencies
~~~~~~~~~~~~~~~~~~

You need to set up packages such as AERIE (for HAWC), or the Fermi
Science Tools, before running the script, otherwise some of the
functionalities will not work.

-  AERIE for HAWC: make sure that this works before running the script:

   .. code:: bash

       > liff-PointSourceExpectation --version
       INFO [CommandLineConfigurator.cc, ParseCommandLine:137]: 

        liff-PointSourceExpectation
        Aerie version: 2.04.00
        Build type: Debug

   If it does not, you need to set up the HAWC environment (refer to the
   appropriate documentation)

-  Fermi Science Tools for Fermi/LAT analysis: make sure that this
   works:
   
   .. code:: bash
       
       > gtirfs   
       ...   
       P8R2_TRANSIENT100_V6::EDISP0   
       P8R2_TRANSIENT100_V6::EDISP1   
       ...
   
   If it does not, you need to configure and set up the Fermi Science
   Tools.

-  ROOT: ROOT is not required by 3ML, but it provides the Minuit2
   minimizer which can be used in 3ML. If you already have ROOT, make sure that
   this works before running the script:
   
   .. code:: bash
       
       > root-config --version
       5.34/36
   
Install using pip (advanced)
----------------------------

Since this method alters the python environment you have on your system,
we suggest you use this method only if you understand the implications.

Remove any previous installation you might have with:

.. code:: bash

    > pip uninstall threeML
    > pip uninstall astromodels
    > pip uninstall cthreeML

then:

.. code:: bash

    > pip install numpy scipy ipython
    > pip install git+https://github.com/giacomov/3ML.git 
    > pip install git+https://github.com/giacomov/astromodels.git --upgrade

In order to use the HAWC plugin, you will also need to install cthreeML
(run this *after* setting up the HAWC environment):

.. code:: bash
    
    > pip install git+https://github.com/giacomov/cthreeML.git

.. _Conda: https://conda.io/docs/
.. _here: https://raw.githubusercontent.com/giacomov/3ML/master/install_3ML.sh
