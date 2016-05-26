import collections
import os
import warnings

import astropy.io.fits as pyfits
import numpy
import scipy.integrate

from threeML.minimizer import minimization
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial
from threeML.plugins.ogip import OGIPPHA
from threeML.plugins.FermiGBMLike import FermiGBMLike
from astromodels.parameter import Parameter

__instrument_name = "Fermi GBM (all detectors)"


class FermiGBMLike(FermiGBMLike):
    def __init__(self, name, ttefile, bkgselections, rspfile):
        '''
        If the input files are TTE files. Background selections are specified as
        a nested list/array e.g. [[-10,0],[10,20]]
        
        FermiGBMLike("GBM","glg_tte_n6_bn080916412.fit",[[-10,0][10,20]],"rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        '''

        self.name = name

        # Check that all file exists
        notExistant = []

        if (not os.path.exists(ttefile):
            notExistant.append(ttefile)


        if (not os.path.exists(rspfile.split("{")[0])):
            notExistant.append(rspfile.split("{")[0])

        if (len(notExistant) > 0):

            for nt in notExistant:
                print("File %s does not exists!" % (nt))

            raise IOError("One or more input file do not exist!")

        self.phafile = OGIPPHA(phafile, filetype='observed')
        self.exposure = self.phafile.getExposure()
        self.bkgfile = OGIPPHA(bkgfile, filetype="background")
        self.response = Response(rspfile)

        # Start with an empty mask (the user will overwrite it using the
        # setActiveMeasurement method)
        self.mask = numpy.asarray(
            numpy.ones(self.phafile.getRates().shape),
            numpy.bool)

        # Get the counts for this spectrum
        self.counts = (self.phafile.getRates()[self.mask]
                       * self.exposure)

        # Check that counts is positive
        idx = (self.counts < 0)

        if (numpy.sum(idx) > 0):
            warnings.warn("The observed spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.counts[idx] = 0

        pass

        # Get the background counts for this spectrum
        self.bkgCounts = (self.bkgfile.getRates()[self.mask]
                          * self.exposure)

        # Check that bkgCounts is positive
        idx = (self.bkgCounts < 0)

        if (numpy.sum(idx) > 0):
            warnings.warn("The background spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.bkgCounts[idx] = 0

        # Check that the observed counts are positive

        idx = self.counts < 0

        if numpy.sum(idx) > 0:
            raise RuntimeError("Negative counts in observed spectrum %s. Data are corrupted." % (phafile))

        # Keep a copy which will never be modified
        self.counts_backup = numpy.array(self.counts, copy=True)
        self.bkgCounts_backup = numpy.array(self.bkgCounts, copy=True)

        # Effective area correction is disabled by default, i.e.,
        # the nuisance parameter is fixed to 1
        self.nuisanceParameters = {}
        self.nuisanceParameters['InterCalib'] = Parameter("InterCalib", 1, min_value=0.9, max_value=1.1, delta=0.01)
        self.nuisanceParameters['InterCalib'].fix = True

    pass

    def _FitBackground(self):

        self._backgroundexists = True
        ## Seperate everything by energy channel
        
        
        eneLcs = []

        eMax = self.chanLU['E_MAX']
        eMin = self.chanLU['E_MIN']
        chanWidth = eMax - eMin

        
        for x,cw in enumerate(chanWidth):

            truthTable = self.evtExt["PHA"] == x

            evts = self.evtExt[truthTable]


            truthTables = []
            for sel in self.bkgIntervals:
                
                truthTables.append(logical_and(evts["TIME"]-self.trigTime>= sel[0] , evts["TIME"]-self.trigTime<= sel[1] ))
                
            
            tt = truthTables[0]
            if len(truthTables)>1:
                                
                for y in truthTables[1:]:
                    
                    tt=logical_or(tt,y)

            self.bkgRegion=tt
            
            evts = evts[tt]

            eneLcs.append(evts)
        self.eneLcs = eneLcs
        self.bkgCoeff = []

        polynomials               = []

      
        for elc,cw in zip(eneLcs,chanWidth):

            cnts,bins=histogram(elc["TIME"]-self.trigTime,bins=self.bins)

 
#            tt=cnts>=0
            meanT=[]
            for i in xrange(len(bins)-1):

                m = mean((bins[i],bins[i+1]))
                meanT.append(m)
            meanT = array(meanT)

            truthTables = []
            for sel in self.bkgIntervals:
                
                truthTables.append(logical_and(meanT>= sel[0] , meanT<= sel[1] ))
                
            
            tt = truthTables[0]
            if len(truthTables)>1:
                                
                for y in truthTables[1:]:
                    
                    tt=logical_or(tt,y)


            
            cnts=cnts/self.binWidth
            
            
            thisPolynomial,cstat    = self._fitChannel(cnts[tt],meanT[tt], self.optimalPolGrade)      


            

            
            polynomials.append(thisPolynomial)
        #pass
        self.polynomials          = polynomials






    
    def useIntercalibrationConst(self, factorLowBound=0.9, factorHiBound=1.1):
        self.nuisanceParameters['InterCalib'].free()
        self.nuisanceParameters['InterCalib'].set_bounds(factorLowBound, factorHiBound)

        # Check that the parameter is within the provided bounds
        value = self.nuisanceParameters['InterCalib'].value

        if (value < factorLowBound):
            warnings.warn(
                "The intercalibration constant was %s, lower than the provided lower bound %s." % (value, factorLowBound) +
                " Setting it equal to the lower bound")

            self.nuisanceParameters['InterCalib'].setValue(float(factorLowBound))

        if (value > factorHiBound):
            warnings.warn(
                "The intercalibration constant was %s, larger than the provided hi bound %s." % (value, factorHiBound) +
                " Setting it equal to the hi bound")

            self.nuisanceParameters['InterCalib'].setValue(float(factorHiBound))















            
    def fixIntercalibrationConst(self, value=None):

        if (value is not None):
            # Fixing the constant to the provided value
            self.nuisanceParameters['InterCalib'].value = float(value)

        else:

            # Do nothing, i.e., leave the constant to the value
            # it currently has
            pass

        self.nuisanceParameters['InterCalib'].fix()

    def setActiveMeasurements(self, *args):
        '''Set the measurements to be used during the analysis.
        Use as many ranges as you need,
        specified as 'emin-emax'. Energies are in keV. Example:

        setActiveMeasurements('10-12.5','56.0-100.0')

        which will set the energy range 10-12.5 keV and 56-100 keV to be
        used in the analysis'''

        # To implelemnt this we will use an array of boolean index,
        # which will filter
        # out the non-used channels during the logLike

        # Now build the mask: values for which the mask is 0 will be masked
        mask = numpy.zeros(self.phafile.getRates().shape)

        for arg in args:
            ee = map(float, arg.replace(" ", "").split("-"))
            emin, emax = sorted(ee)
            idx1 = self.response.energyToChannel(emin)
            idx2 = self.response.energyToChannel(emax)
            mask[idx1:idx2 + 1] = True
        pass
        self.mask = numpy.array(mask, numpy.bool)

        self.counts = self.counts_backup[self.mask]
        self.bkgCounts = self.bkgCounts_backup[self.mask]

        print("Now using %s channels out of %s" % (numpy.sum(self.mask),
                                                   self.phafile.getRates().shape[0]
                                                   ))

    pass

    def get_name(self):
        '''
        Return a name for this dataset (likely set during the constructor)
        '''
        return self.name

    pass

    def set_model(self, likelihoodModel):
        '''
        Set the model to be used in the joint minimization.
        '''
        self.likelihoodModel = likelihoodModel

        nPointSources = self.likelihoodModel.get_number_of_point_sources()

        # This is a wrapper which iterates over all the point sources and get
        # the fluxes
        # We assume there are no extended sources, since the GBM cannot handle them

        def diffFlux(energies):
            fluxes = self.likelihoodModel.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, nPointSources):
                fluxes += self.likelihoodModel.get_point_source_fluxes(i, energies)

            return fluxes

        self.diffFlux = diffFlux

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):
            # Simpson's rule

            return (e2 - e1) / 6.0 * (self.diffFlux(e1)
                                      + 4 * self.diffFlux((e1 + e2) / 2.0)
                                      + self.diffFlux(e2))

        self.response.setFunction(diffFlux,
                                  integral)

    pass

    def inner_fit(self):

        # Effective area correction
        if (self.nuisanceParameters['InterCalib'].free):

            # A true fit would be an overkill, and slow
            # Just sample a 100 values and choose the minimum
            values = numpy.linspace(self.nuisanceParameters['InterCalib'].min_value,
                                    self.nuisanceParameters['InterCalib'].max_value,
                                    100)

            # I do not use getLogLike so I can compute only once the folded model
            # (which is not going to change during the inner fit)

            folded = self.getFoldedModel()

            modelCounts = folded * self.exposure

            def fitfun(cons):

                self.nuisanceParameters['InterCalib'].value = cons

                return (-1) * self._computeLogLike(
                    self.nuisanceParameters['InterCalib'].value * modelCounts + self.bkgCounts)

            logLval = map(fitfun, values)
            idx = numpy.argmax(logLval)
            self.nuisanceParameters['InterCalib'].value = values[idx]
            # return logLval[idx]

            # Now refine with minuit

            parameters = collections.OrderedDict()
            parameters[(self.name, 'InterCalib')] = self.nuisanceParameters['InterCalib']
            minimizer = minimization.MinuitMinimizer(fitfun, parameters)
            bestFit, mlogLmin = minimizer.minimize()

            return mlogLmin * (-1)

        else:

            return self.get_log_like()

    def getFoldedModel(self):

        # Get the folded model for this spectrum
        # (this is the rate predicted, in cts/s)

        return self.response.convolve()[self.mask]

    def getModelAndData(self):

        e1, e2 = (self.response.ebounds[:, 0],
                  self.response.ebounds[:, 1])

        return (self.response.convolve()[self.mask] * self.exposure
                + self.bkgCounts,
                e1[self.mask],
                e2[self.mask],
                self.counts)

    def _getModelCounts(self):

        # Get the folded model for this spectrum (this is the rate predicted,
        # in cts/s)

        folded = self.getFoldedModel()

        # Model is folded+background (i.e., we assume negligible errors on the
        # background)
        modelCounts = self.nuisanceParameters['InterCalib'].value * folded * self.exposure + self.bkgCounts

        return modelCounts

    def _computeLogLike(self, modelCounts):

        return numpy.sum(- modelCounts
                         + self.counts * numpy.log(modelCounts)
                         - logfactorial(self.counts))

    def get_log_like(self):
        '''
        Return the value of the log-likelihood with the current values for the
        parameters
        '''

        modelCounts = self._getModelCounts()

        logLike = self._computeLogLike(modelCounts)

        return logLike

    def get_nuisance_parameters(self):
        '''
        Return a list of nuisance parameter names. Return an empty list if there
        are no nuisance parameters
        '''
        return self.nuisanceParameters.keys()

    pass


pass

class GBMTTEFile(object):

    def __init__(ttefile):

        tte = pyfits.open(ttefile)
        
        self._events = tte['EVENTS'].data['TIME']
        self._phatag = tte['EVENTS'].data['PHA']
        self._triggertime = tte['PRIMARY'].header['TRIGTIME']
        self._startevents = tte['PRIMARY'].header['TSTART']
        self._stopevents = tte['PRIMARY'].header['TSTOP']
        self.nchans = tte['EBOUNDS']['NAXIS2']


