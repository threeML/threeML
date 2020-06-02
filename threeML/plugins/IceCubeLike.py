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


r"""This IceCube plugin is currently under develop by Kwok Lung Fan and John Evans"""


class SpectrumConverter(object):
    r"""
    A class that converter a astromodels model instance to a spectrum object with __call__ method.
    """
    def __init__(self,likelihood_model_instance):
        r"""Constructor of the class"""
        self.__model=likelihood_model_instance
        
    def __call__(self, E, **kwargs):
        r"""Evaluate spectrum at E """
        return self.__model.get_point_source_fluxes(id=0,energies=E)
    
    #Function below exists only to avoid skylab conflict and can be removed after get rid of skylab dependence. Not sure removing them cause any problem now
    def validate(self):
        pass

    def __str__(self):
        r"""String representation of class"""
        return "SpectrumConverter class doesn't support string representation now"

    def copy(self):
        r"""Return copy of this class"""
        c = type(self).__new__(type(self))
        c.__dict__.update(self.__dict__)
        return c
        
class WeightClassicLLH(ClassicLLH):

    def add_spectrum(self,spectrum):
        self.__spectrum=spectrum
        
    def weight(self, ev, **params):
        r"""This part is wrong. The correct way is to multiply by the oneweight spline.
        """
        if self.__spectrum is not None:
            return self.__spectrum(np.exp(ev['logE'])),None
        else:
            return np.ones(len(ev)),None

        

class IceCubeLike(PluginPrototype):
    def __init__(self, name, exp, mc, livetime,sinDec_bins=np.linspace(-np.pi/2,np.pi/2,30),**kwargs):
        r"""Constructor of the class.
        exp is the data,mc is the monte carlo,livetime is the livetime of the data.
        The default sin dec bins is set but can be pass by sinDec_bins.
        The current version doesn't support a time dependent search.
        """
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
        r"""Setting up the model"""
        if likelihood_model_instance is None:

            return

        if likelihood_model_instance.point_sources is not None:
            assert (
                likelihood_model_instance.get_number_of_extended_sources() == 0
            ), "Current IceCubeLike does not support extended sources"
            assert (
                likelihood_model_instance.get_number_of_point_sources() == 1
            ), "Current IceCubeLike does not support more than one point source"
            #extracting the point source location
            ra,dec=likelihood_model_instance.get_point_source_position(id=0)
            self._ra=ra*np.pi/180 #convert it to radian
            self._dec=dec*np.pi/180 #convert it to radian
            
            self._llh_instance=likelihood_model_instance
            self._spectrum=SpectrumConverter(likelihood_model_instance)
            self._llh_model=WeightClassicLLH(sinDec_bins=self._sinDec_bins)
            self._llh_model.add_spectrum(self._spectrum)
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
        return llh[0]
        
    def inner_fit(self):
        return self.get_log_like()