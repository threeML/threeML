from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import collections
import os
import sys
from copy import deepcopy
import numpy as np
from astromodels import Parameter
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.plugin_prototype import PluginPrototype

from mla.core import *
from mla.spectral import *
__instrument_name = "IceCube"


r"""This IceCube plugin is currently under develop by Kwok Lung Fan and John Evans"""


class Spectrum(object):
    r"""
    A class that converter a astromodels model instance to a spectrum object with __call__ method.
    """
    def __init__(self,likelihood_model_instance):
        r"""Constructor of the class"""
        self.model=likelihood_model_instance
        
    def __call__(self, E, **kwargs):
        r"""Evaluate spectrum at E """
        return self.model.get_particle_source_fluxes(id=0,energies=E)
    
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
        

class IceCubeLike(PluginPrototype):
    def __init__(self, name, exp, mc, livetime,**kwargs):
        r"""Constructor of the class.
        exp is the data,mc is the monte carlo,livetime is the livetime of the data.
        The default sin dec bins is set but can be pass by sinDec_bins.
        The current version doesn't support a time dependent search.
        """
        nuisance_parameters = {}
        super(IceCubeLike, self).__init__(name, nuisance_parameters)
        self.parameter=kwargs
        self.data=exp
        self.mc=mc
        self.livetime=livetime
        self.llh_model = None
        self.sample=None
       
       
        
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
            self.ra=ra*np.pi/180 #convert it to radian
            self.dec=dec*np.pi/180 #convert it to radian
            self.source_name=likelihood_model_instance.get_point_source_name(id=0)
            self.llh_instance=likelihood_model_instance
            self.spectrum=Spectrum(likelihood_model_instance)
            self.llh_model=LLH_point_source(self.ra , self.dec ,  self.data , self.mc , self.livetime , self.spectrum) 
            if self.sample != None:
                self.llh_model.add_injection(self.sample)
                

        else:
            print("No point sources in the model")
            return
            
    def overload_llh(self,llh_model):
        self.llh_model=llh_model
        return
        
        
    def update_model(self):            
        self.spectrum=Spectrum(self.llh_instance)
        return
    
    def add_injection(self,sample):
        self.sample=sample
        return
    
    def get_log_like(self):
        self.update_model()
        llh=self.llh_model.eval_llh()
        print(llh)
        return llh[1]
        
    def inner_fit(self):
        return self.get_log_like()