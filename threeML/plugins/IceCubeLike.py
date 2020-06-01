from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import collections
import os
import sys

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astromodels import Parameter
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.plugin_prototype import PluginPrototype
from skylab import *
from skylab.llh_models import *
from skylab.ps_llh import *
from skylab.spectral_models import *
__instrument_name = "IceCube"

class SpectrumConverter(BaseSpectrum):
    def __init__(self,likelihood_model_instance):
        self.__model=likelihood_model_instance
    def __call__(self, E, **kwargs):
        return self.__model.get_point_source_fluxes(id=0,energies=E)

        

class IceCubeLike(PluginPrototype):

    def __init__(self, name, exp, mc, livetime,sinDec_bins=np.linspace(-np.pi/2,np.pi/2,30),**kwargs):
        nuisance_parameters = {}
        super(IceCubeLike, self).__init__(name, nuisance_parameters)
        self._parameter=kwargs
        self._exp=exp
        self._mc=mc
        self._sinDec_bins=sinDec_bins
        self._livetime=livetime
        self._llh_model = None
        self._sample=None
        
    def set_model(self, likelihood_model_instance):
        if likelihood_model_instance is None:

            return

        if likelihood_model_instance.point_sources is not None:
            assert (
                likelihood_model_instance.get_number_of_extended_sources() == 0
            ), "Current IceCubeLike does not support extended sources"
            assert (
                likelihood_model_instance.get_number_of_point_sources() == 1
            ), "Current IceCubeLike does not support more than one point source"
            ra,dec=likelihood_model_instance.get_point_source_position(id=0)
            self._ra=ra*np.pi/180
            self._dec=dec*np.pi/180
            self._llh_instance=likelihood_model_instance
            self._spectrum=SpectrumConverter(likelihood_model_instance)
            self._llh_model=ClassicLLH(spectrum=self._spectrum,sinDec_bins=self._sinDec_bins)
            self._ps_llh=PointSourceLLH(self._exp, self._mc, self._livetime,self._llh_model,self._parameter)
            self._ps_llh._select_events(np.array([self._ra]),np.array([self._dec]),np.array([1]),inject=self._sample)
        else:
            print("No point sources in the model")
            return
            
    def overload_llh(self,llh_model):
        self._llh_model=llh_model
        self._ps_llh.update_model(self._llh_model,self._exp)
        #self._ps_llh._select_events(self._ra,self._dec)
        return
        
    def update_model(self):
        self._spectrum=SpectrumConverter(self._llh_instance)
        self._llh_model._effA(self._mc,self._livetime,spectrum=self._spectrum)
        self.overload_llh(self._llh_model)
        return
    
    def add_injection(self,sample):
        self._sample=sample
        return
    
    def get_log_like(self):
        self.update_model()
        nsig=self._llh_model.effA(self._dec)[0]
        llh=self._ps_llh.llh(nsignal=nsig)
        print(nsig,llh[0])
        #self._ps_llh._reset_background_pdf()
        return llh[0]
        
    def inner_fit(self):
        return self.get_log_like()