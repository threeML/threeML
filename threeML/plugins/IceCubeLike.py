from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import collections
import os
import sys
from copy import deepcopy,copy
import numpy as np
from astromodels import Parameter
from astromodels.core.sky_direction import SkyDirection
from astromodels.core.spectral_component import SpectralComponent
from astromodels.core.tree import Node
from astromodels.core.units import get_units
from astromodels.sources.source import Source, POINT_SOURCE
from astromodels.utils.pretty_list import dict_to_list
from astromodels.core.memoization import use_astromodels_memoization
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
from threeML.plugin_prototype import PluginPrototype
from threeML.classicMLE.joint_likelihood import JointLikelihood
from astromodels import PointSource,ExtendedSource
import astropy.units as u
from mla.core import *
from mla.spectral import *
from mla.injection import *
__instrument_name = "IceCube"

__all__=["NeutrinoPointSource","NeutrinoExtendedSource"]
r"""This IceCube plugin is currently under develop by Kwok Lung Fan and John Evans"""
class NeutrinoPointSource(PointSource):
    def __init__(self, source_name, ra=None, dec=None, spectral_shape=None,
                 l=None, b=None, components=None, sky_position=None):

        # Check that we have all the required information

        # (the '^' operator acts as XOR on booleans)

        # Check that we have one and only one specification of the position

        assert ((ra is not None and dec is not None) ^
                (l is not None and b is not None) ^
                (sky_position is not None)), "You have to provide one and only one specification for the position"

        # Gather the position

        if not isinstance(sky_position, SkyDirection):

            if (ra is not None) and (dec is not None):

                # Check that ra and dec are actually numbers

                try:

                    ra = float(ra)
                    dec = float(dec)

                except (TypeError, ValueError):

                    raise AssertionError("RA and Dec must be numbers. If you are confused by this message, you "
                                         "are likely using the constructor in the wrong way. Check the documentation.")

                sky_position = SkyDirection(ra=ra, dec=dec)

            else:

                sky_position = SkyDirection(l=l, b=b)

        self._sky_position = sky_position

        # Now gather the component(s)

        # We need either a single component, or a list of components, but not both
        # (that's the ^ symbol)

        assert (spectral_shape is not None) ^ (components is not None), "You have to provide either a single " \
                                                                        "component, or a list of components " \
                                                                        "(but not both)."

        # If the user specified only one component, make a list of one element with a default name ("main")

        if spectral_shape is not None:

            components = [SpectralComponent("main", spectral_shape)]

        Source.__init__(self, components, POINT_SOURCE)

        # A source is also a Node in the tree

        Node.__init__(self, source_name)

        # Add the position as a child node, with an explicit name

        self._add_child(self._sky_position)

        # Add a node called 'spectrum'

        spectrum_node = Node('spectrum')
        spectrum_node._add_children(list(self._components.values()))

        self._add_child(spectrum_node)

        # Now set the units
        # Now sets the units of the parameters for the energy domain

        current_units = get_units()

        # Components in this case have energy as x and differential flux as y

        x_unit = current_units.energy
        y_unit = (current_units.energy * current_units.area * current_units.time) ** (-1)

        # Now set the units of the components
        for component in list(self._components.values()):

            component.shape.set_units(x_unit, y_unit)
    def __call__(self, x, tag=None):
        if isinstance(x, u.Quantity):
            if type(x) == float or type(x) == int:
                return 0*(u.keV**-1*u.cm**-2*u.second**-1)
            return np.zeros((len(x)))*(u.keV**-1*u.cm**-2*u.second**-1) #It is zero so the unit doesn't matter
        else:
            if type(x) == float or type(x) == int:
                return 0
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
    def __init__(self,likelihood_model_instance,A = 1):
        r"""Constructor of the class"""
        self.model=likelihood_model_instance
        self.A = A
        for source_name, source in likelihood_model_instance._point_sources.items():
            if isinstance(source,NeutrinoPointSource):
                self.neutrinopointsource = source_name
        
    def __call__(self, E, **kwargs):
        r"""Evaluate spectrum at E """
        return self.model._point_sources[self.neutrinopointsource].call(E)*self.A
    
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
    def __init__(self, name, exp, mc,background, signal_time_profile = None , background_time_profile = (50000,70000) , fit_position=False,verbose=False,**kwargs):
        r"""Constructor of the class.
        exp is the data,mc is the monte carlo
        The default sin dec bins is set but can be pass by sinDec_bins.
        The current version doesn't support a time dependent search.
        """
        nuisance_parameters = {}
        super(IceCubeLike, self).__init__(name, nuisance_parameters)
        self.parameter=kwargs
        self.llh_model = None
        self.sample = None
        self.fit_position = fit_position
        self.spectrum = PowerLaw(1,1e-15,-2)
        self.verbose = verbose
        if isinstance(background_time_profile,generic_profile):
            pass
        else:
            background_time_profile = uniform_profile(background_time_profile[0],background_time_profile[1])
        if signal_time_profile is None:
            signal_time_profile = deepcopy(background_time_profile)
        else:
            signal_time_profile = signal_time_profile
            
        self.llh_model=LLH_point_source(ra=np.pi/2 , dec=np.pi/6 , data = exp , sim = mc ,spectrum = self.spectrum ,signal_time_profile = signal_time_profile , background_time_profile = background_time_profile, background = background ,fit_position=True) 
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
                    self.llh_model.update_time_weight()
                    self.llh_model.update_energy_weight()
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
        if self.verbose:
            print(llh)
        return llh[1]
        
    def inner_fit(self):
        return self.get_log_like()

class sensitivity(object):

    def __init__(self,JointLikelihood_instance):
        self.jl = JointLikelihood_instance
        for i in JointLikelihood_instance._data_list.keys():
            if isinstance(JointLikelihood_instance._data_list[i],IceCubeLike):
                self.jl_key = i
                self.jl_value = JointLikelihood_instance._data_list[i]
                self.jl_value.verbose = False
        
        self.list_of_source = deepcopy(self.jl._likelihood_model.sources)
        return 
    
    
    def set_backround(self, grl, background_model ,time_window = None ,background_window = 14 , start_time = None):
        r'''Setting the background information which will later be used when drawing data as background
        args:
        grl:The good run list
        background_window: The time window(days) that will be used to estimated the background rate and drawn sample from.Default is 14 days
        '''
        if start_time is None:
            start_time = self.jl_value.llh_model.background_time_profile.get_range()[0]
        fully_contained = (grl['start'] >= start_time-background_window) &\
                            (grl['stop'] < start_time)
        start_contained = (grl['start'] < start_time-background_window) &\
                            (grl['stop'] > start_time-background_window)
        background_runs = (fully_contained | start_contained)
        if not np.any(background_runs):
            print("ERROR: No runs found in GRL for calculation of "
                  "background rates!")
            raise RuntimeError
        background_grl = grl[background_runs]
        if time_window is None:
            time_window = self.jl_value.llh_model.background_time_profile.effective_exposure()
        # Get the number of events we see from these runs and scale 
        # it to the number we expect for our search livetime.
        n_background = background_grl['events'].sum()
        n_background /= background_grl['livetime'].sum()
        n_background *= time_window
        self.n_background = n_background
        self.background_model = background_model
        return    
        
    def draw_data(self):
        r'''Draw data sample
        return:
        background: background sample
        '''
        n_background_observed = np.random.poisson(self.n_background)
        background = np.random.choice(self.jl_value.llh_model.background, n_background_observed).copy()
        background['time'] = self.jl_value.llh_model.background_time_profile.random(len(background))
        return background
    
    def set_injection( self,signal_time_profile = None , background_time_profile = (50000,70000), sampling_width = np.radians(1) ):
        r'''Set the details of the injection.
        sim: Simulation data
        gamma: Spectral index of the injection spectrum
        signal_time_profile: generic_profile object. This is the signal time profile.Default is the same as background_time_profile.
        background_time_profile: generic_profile object or the list of the start time and end time. This is the background time profile.Default is a (0,1) tuple which will create a uniform_profile from 0 to 1.
        '''
        spectrum = PowerLaw(1,1e-15,-2)
        self.PS_injector = PSinjector(spectrum, self.jl_value.llh_model.fullsim , signal_time_profile = None , background_time_profile = background_time_profile)
        self.PS_injector.set_source_location(self.jl_value.ra,self.jl_value.dec,sampling_width = sampling_width)
        return
    
    def draw_signal(self):
        r'''Draw signal sample
        return:
        signal: signal sample
        '''
        return self.PS_injector.sample_from_spectrum()
    
    def build_background_TS(self,n_trials = 1000):
        r'''build background TS distribution
        args:
        n_trials: Number of trials
        return:
        TS: The TS array
        '''
        TS = []
        print("start building Background TS")
        for i in range(n_trials):
            print(i)
            for source_name,source in self.list_of_source.items():
                self.jl._likelihood_model.remove_source(source_name)
            for source_name,source in self.list_of_source.items():
                self.jl._likelihood_model.add_source(deepcopy(source))
            self.jl_value.llh_model.update_data(self.draw_data())
            try:
                self.jl.fit(quiet=True)
            except OverflowError:
                pass
            except IndexError:
                pass
            TS.append(self.jl._current_minimum)
        return np.array(TS)
    
    def build_signal_TS(self, signal_trials = 200):
        r'''build signal TS distribution
        args:
        signal_trials: Number of trials
        result: Whether storing the full result in self.result.Default is False.
        result_file:Whether storing the full result in file.Default is False.
        
        return:
        TS: The TS array
        '''
        TS = []
        for i in range(signal_trials):
            data = self.draw_data()
            signal = self.draw_signal()
            signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
            if not n in data.dtype.names])
            self.jl_value.llh_model.update_data(np.concatenate([data,signal]))
            try:
                #self.jl_value.verbose=True
                self.jl.fit(quiet=True)
            except OverflowError:
                pass
            except IndexError:
                pass
            TS.append(-self.jl._current_minimum)
        return np.array(TS)
        
    def calculate_ratio_passthreshold(self, signal_trials = 200):
        r'''Calculate the ratio of signal trials passing the threshold
        args:
        signal_trials: Number of signal trials
        result: Whether storing the full result in self.result.Default is False.
        result_file:Whether storing the full result in file.Default is False.
        
        return:
        result:The ratio of passing(both for three sigma and median of the background
        '''
        signal_ts = self.build_signal_TS(signal_trials)
        result = [(signal_ts > self.bkg_three_sigma ).sum()/float(len(signal_ts)), (signal_ts > self.bkg_median).sum()/float(len(signal_ts))]
        return result
        
    def calculate_sensitivity(self, base_spectrum, bkg_trials = 1000, signal_trials = 200,list_N = [1] ,N_factor = 1.5 , make_plot = None ,Threshold_list=[90] , Threshold_potential = [50]):
        r'''Calculate the sensitivity plus the discovery potential
        args:
        base_spectrum = The spectrum
        bkg_trials : Number of background trials
        signal_trials: Number of signal trials
        list_N:The list of flux norm to test and build the spline
        N_factor: Factor for Flux increments .If the maximum in list_N still wasn't enough to pass the threshold, the program will enter a while loop with N_factor*N tested each times until the N passed the threshold.
        make_plot: The file name of the plot saved. Default is not saving
        Threshold_list: The list of threshold of signal TS passing Median of the background TS. 
        Threshold_potential: The list of threshold of signal TS passing 3 sigma of the background TS. 
        '''
        self.Threshold_list = Threshold_list
        self.base_spectrum = base_spectrum
        self.base_spectrum.A = list_N[0]
        self.Threshold_potential = Threshold_potential
        max_threshold = np.array(Threshold_list).max()
        max_potential = np.array(Threshold_potential).max()
        list_N = np.array(deepcopy(list_N))
        result = []
        self.ts_bkg = self.build_background_TS(bkg_trials)
        self.bkg_median = np.percentile(self.ts_bkg , 50)
        self.bkg_three_sigma = np.percentile(self.ts_bkg , 99.7)
        for N in list_N:
            print("Now testing factor: "+ str(N))
            self.base_spectrum.A = N
            self.PS_injector.update_spectrum(self.base_spectrum)
            tempresult = self.calculate_ratio_passthreshold( signal_trials = 200)
            print(tempresult)
            result.append(tempresult)
        if tempresult[0] < max_potential*0.01 or tempresult[1] < max_threshold*0.01:
            reach_max = False
            N = N * N_factor
            list_N = np.append(list_N,N)
        else:
            reach_max = True
        while not reach_max:
            print("Now testing : "+ str(N))
            self.base_spectrum.factor = N
            self.PS_injector.update_spectrum(self.base_spectrum)
            tempresult = self.calculate_ratio_passthreshold(bkg_trials = 1000, signal_trials = 200)
            print(tempresult)
            result.append(tempresult)
            if tempresult[0] < max_potential*0.01 or tempresult[1] < max_threshold*0.01:
                N = N * N_factor
                list_N = np.append(list_N,N)
            else:
                reach_max = True
        result = np.array(result)
        self.result = result
        self.list_N = list_N
        self.spline_sigma = interpolate.UnivariateSpline(list_N,result[:,0] , ext = 3)
        self.spline_sen = interpolate.UnivariateSpline( list_N,result[:,1] , ext = 3)
        Threshold_result = []
        Threshold_potential_result = []
        for i in Threshold_list:
            tempspline = interpolate.UnivariateSpline(list_N,result[:,1]-i*0.01 , ext = 3)
            Threshold_result.append(tempspline.roots()[0])
            print("Threshold: " + str(i) + ", N : " + str(self.spline_sen(i*0.01)))
        for i in Threshold_potential:
            tempspline = interpolate.UnivariateSpline(list_N,result[:,0]-i*0.01 , ext = 3)
            Threshold_potential_result.append(tempspline.roots()[0])
            print("Threshold_potential: " + str(i) + ", N : " + str(self.spline_sigma(i*0.01)))    
        self.Threshold_result = Threshold_result
        self.Threshold_potential_result = Threshold_potential_result
        if make_plot != None :
           self.make_plot(make_plot)
        return

    def make_plot(self,file_name):
        r'''save plot to file_name
        '''
        fig, ax = plt.subplots(figsize = (12,12))
        ax.scatter(self.list_N,self.result[:,1],label = 'sensitiviy point',color='r')
        ax.scatter(self.list_N,self.result[:,0],label = 'potential point',color='b')
        ax.set_xlim(self.list_N[0],self.list_N[-1])
        ax.plot(np.linspace(self.list_N[0],self.list_N[-1],1000),self.spline_sen(np.linspace(self.list_N[0],self.list_N[-1],1000)),label = 'sensitiviy spline',color='r')
        ax.plot(np.linspace(self.list_N[0],self.list_N[-1],1000),self.spline_sigma(np.linspace(self.list_N[0],self.list_N[-1],1000)),label = 'potential spline',color='b')
        for i in range(len(self.Threshold_result)):
            ax.axvline(self.Threshold_result[i],label = 'sensitiviy '+str(self.Threshold_list[i]),color='r')
        for i in range(len(self.Threshold_potential_result)):
            ax.axvline(self.Threshold_potential_result[i],label = 'potential '+str(self.Threshold_potential[i]),color='b')
        ax.set_title("Flux norm vs passing ratio",fontsize=14)
        ax.set_xlabel(r"Flux Norm($GeV cm^{-2} s^{-1}$)",fontsize=14)
        ax.set_ylabel(r"Passing ratio",fontsize=14)
        ax.legend(fontsize=14)
        fig.savefig(file_name)
        plt.close()        