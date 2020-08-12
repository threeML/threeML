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
from astromodels import PointSource,ExtendedSource
import astropy.units as u
from mla.core import *
from mla.spectral import *
__instrument_name = "IceCube"

__all__=["NeutrinoPointSource","NeutrinoExtendedSource"]
r"""This IceCube plugin is currently under develop by Kwok Lung Fan and John Evans"""
class NeutrinoPointSource(PointSource):
    def __call__(self, x, tag=None):
        if isinstance(x, u.Quantity):
            return np.zeros((len(x)))*(u.keV**-1*u.cm**-2*u.second**-1) #It is zero so the unit doesn't matter
        else:
            return np.zeros((len(x)))
            
    def call(self, x, tag=None):
        if tag is None:

            # No integration nor time-varying or whatever-varying

            if isinstance(x, u.Quantity):

                # Slow version with units

                results = [component.shape(x) for component in list(self.components.values())]

                # We need to sum like this (slower) because using np.sum will not preserve the units
                # (thanks astropy.units)

                return sum(results)

            else:

                # Fast version without units, where x is supposed to be in the same units as currently defined in
                # units.get_units()

                results = [component.shape(x) for component in list(self.components.values())]

                return np.sum(results, 0)

        else:

            # Time-varying or energy-varying or whatever-varying

            integration_variable, a, b = tag

            if b is None:

                # Evaluate in a, do not integrate

                # Suspend memoization because the memoization gets confused when integrating
                with use_astromodels_memoization(False):

                    integration_variable.value = a

                    res = self.__call__(x, tag=None)

                return res

            else:

                # Integrate between a and b

                integrals = np.zeros(len(x))

                # TODO: implement an integration scheme avoiding the for loop

                # Suspend memoization because the memoization gets confused when integrating
                with use_astromodels_memoization(False):

                    reentrant_call = self.__call__

                    for i, e in enumerate(x):

                        def integral(y):

                            integration_variable.value = y

                            return reentrant_call(e, tag=None)

                        # Now integrate
                        integrals[i] = scipy.integrate.quad(integral, a, b, epsrel=1e-5)[0]

                return old_div(integrals, (b - a))

class NeutrinoExtendedSource(ExtendedSource):
    pass

class Spectrum(object):
    r"""
    A class that converter a astromodels model instance to a spectrum object with __call__ method.
    """
    def __init__(self,likelihood_model_instance):
        r"""Constructor of the class"""
        self.model=likelihood_model_instance
        for source_name, source in likelihood_model_instance._point_sources.items():
            if isinstance(source,NeutrinoPointSource):
                self.neutrinopointsource = source_name
        
    def __call__(self, E, **kwargs):
        r"""Evaluate spectrum at E """
        return self.model._point_sources[self.neutrinopointsource].call(E)
    
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
    def __init__(self, name, exp, mc,background, signal_time_profile = None , background_time_profile = (50000,70000) , fit_position=False ,**kwargs):
        r"""Constructor of the class.
        exp is the data,mc is the monte carlo
        The default sin dec bins is set but can be pass by sinDec_bins.
        The current version doesn't support a time dependent search.
        """
        nuisance_parameters = {}
        super(IceCubeLike, self).__init__(name, nuisance_parameters)
        self.parameter=kwargs
        self.data = exp
        self.mc = mc
        self.llh_model = None
        self.sample = None
        self.fit_position = fit_position
        self.spectrum = PowerLaw(1,1e-15,-2)
        if isinstance(background_time_profile,generic_profile):
            pass
        else:
            background_time_profile = uniform_profile(background_time_profile[0],background_time_profile[1])
        if signal_time_profile is None:
            signal_time_profile = deepcopy(background_time_profile)
        else:
            signal_time_profile = signal_time_profile
            
        self.llh_model=LLH_point_source(ra=np.pi/2 , dec=np.pi/6 , data = self.data , sim = self.mc ,spectrum = self.spectrum ,signal_time_profile = signal_time_profile , background_time_profile = background_time_profile, background = background ,fit_position=False) 
        self.dec = None
        self.ra = None
        
    def set_model(self, likelihood_model_instance ):
        r"""Setting up the model"""
        if likelihood_model_instance is None:

            return

        for source_name, source in likelihood_model_instance._point_sources.items():
            if isinstance(source,NeutrinoPointSource):
                self.source_name = source_name
                ra = source.position.get_ra()
                dec = source.position.get_dec()
                ra = ra*np.pi/180 #convert it to radian
                dec = dec*np.pi/180 #convert it to radian
                if self.ra == ra and self.dec == dec :
                    pass
                else:
                    self.ra = ra
                    self.dec = dec
                    self.llh_model.fit_position = self.fit_position
                    self.llh_model.update_position(ra,dec, sampling_width = np.radians(1))
                
            self.model=likelihood_model_instance
            #self.source_name=likelihood_model_instance.get_point_source_name(id=0)
            #self.spectrum=Spectrum(likelihood_model_instance)
            #self.llh_model.update_spectrum(self.spectrum)
                

        if self.source_name is None:
            print("No point sources in the model")
            return
            
    def overload_llh(self,llh_model):
        self.llh_model=llh_model
        return
        
        
    def update_model(self):   
        self.spectrum=Spectrum(self.model)
        self.llh_model.update_spectrum(self.spectrum)    
        #self.llh_model.update_energy_weight()
        return
    
    def add_injection(self,sample):
        self.llh_model.add_injection(sample)
        return
    
    def get_log_like(self):
        self.update_model()
        llh=self.llh_model.eval_llh()
        print(llh)
        return llh[1]
        
    def inner_fit(self):
        return self.get_log_like()