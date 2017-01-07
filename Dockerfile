# This is for the mybinder.org container

FROM andrewosh/binder-base

MAINTAINER Giacomo Vianello <giacomov@stanford.edu>

#USER root
#
#RUN apt-get update
#
#RUN pip install -I --upgrade setuptools
#RUN pip install --upgrade ipywidgets
#RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

USER main

RUN pip install git+https://github.com/giacomov/3ML.git && pip install git+https://github.com/giacomov/astromodels.git

WORKDIR /home/main/notebooks/examples/
