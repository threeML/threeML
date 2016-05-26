import collections
import os
import warnings

import astropy.io.fits as pyfits
import numpy as np
import scipy.integrate
from math import log

from threeML.minimizer import minimization
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.gammaln import logfactorial
from threeML.plugins.ogip import OGIPPHA

from astromodels.parameter import Parameter

__instrument_name = "Fermi GBM (all detectors)"


class FermiGBMLike(PluginPrototype):
    def __init__(self, name, phafile, bkgfile, rspfile):
        '''
        If the input files are PHA2 files, remember to specify the spectrum number, for example:
        FermiGBMLike("GBM","spectrum.pha{2}","bkgfile.bkg{2}","rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        '''

        self.name = name

        # Check that all file exists
        notExistant = []

        if (not os.path.exists(phafile.split("{")[0])):
            notExistant.append(phafile.split("{")[0])

        if (not os.path.exists(bkgfile.split("{")[0])):
            notExistant.append(bkgfile.split("{")[0])

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
        self.mask = np.asarray(
            np.ones(self.phafile.getRates().shape),
            np.bool)

        # Get the counts for this spectrum
        self.counts = (self.phafile.getRates()[self.mask]
                       * self.exposure)


        # Check that counts is positive
        idx = (self.counts < 0)

        if (np.sum(idx) > 0):
            warnings.warn("The observed spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.counts[idx] = 0

        # Get the background counts for this spectrum
        self.bkgCounts = (self.bkgfile.getRates()[self.mask]
                          * self.exposure)

        # Get the error on the background counts
        self.bkgErr = self.bkgfile.getRatesErrors()[self.mask] * self.exposure

        # Check that bkgCounts is positive
        idx = (self.bkgCounts < 0)

        if (np.sum(idx) > 0):
            warnings.warn("The background spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.bkgCounts[idx] = 0

        # Check that bkgErr is positive
        idx = (self.bkgCounts < 0)

        if (np.sum(idx) > 0):

            raise RuntimeError("The background spectrum for %s " % self.name +
                                "has negative errors! Fixing those to zero.")

        # Check that errors are zeros when the bkg is zero
        if np.any((self.bkgErr == 0) & (self.bkgCounts > 0)):

            raise RuntimeError("The background error is zero but the background counts are not zero for some bins. "
                               "Data are corrupted.")

        # Check that the observed counts are positive

        idx = self.counts < 0

        if np.sum(idx) > 0:
            raise RuntimeError("Negative counts in observed spectrum %s. Data are corrupted." % (phafile))

        # Keep a copy which will never be modified
        self.counts_backup = np.array(self.counts, copy=True)
        self.bkgCounts_backup = np.array(self.bkgCounts, copy=True)
        self.bkgErr_backup = np.array(self.bkgErr, copy=True)

        # Effective area correction is disabled by default, i.e.,
        # the nuisance parameter is fixed to 1
        self.nuisanceParameters = {}
        self.nuisanceParameters['InterCalib'] = Parameter("InterCalib", 1, min_value=0.9, max_value=1.1, delta=0.01)
        self.nuisanceParameters['InterCalib'].fix = True

    pass

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
        mask = np.zeros(self.phafile.getRates().shape)

        for arg in args:
            ee = map(float, arg.replace(" ", "").split("-"))
            emin, emax = sorted(ee)
            idx1 = self.response.energyToChannel(emin)
            idx2 = self.response.energyToChannel(emax)
            mask[idx1:idx2 + 1] = True
        pass
        self.mask = np.array(mask, np.bool)

        self.counts = self.counts_backup[self.mask]
        self.bkgCounts = self.bkgCounts_backup[self.mask]
        self.bkgErr = self.bkgErr_backup[self.mask]

        print("Now using %s channels out of %s" % (np.sum(self.mask),
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
        modelCounts = self.nuisanceParameters['InterCalib'].value * folded * self.exposure

        # Put all negative predicted counts to zero
        clippedModelCounts = np.clip(modelCounts, 0, 1e10)

        return clippedModelCounts

    def _computeLogLike(self, modelCounts):

        # This loglike assume Gaussian errors on the background and Poisson uncertainties on the
        # observed counts. It is a profile likelihood.

        MB = self.bkgCounts + modelCounts
        s2 = self.bkgErr**2

        b = 0.5 * (np.sqrt( MB**2 - 2 * s2 * (MB - 2 * self.counts) + self.bkgErr**4)
                   + self.bkgCounts - modelCounts - s2)

        # Now there are two branches: when the background is 0 we are in the normal situation of a pure
        # Poisson likelihood, while when the background is not zero we use the profile likelihood

        # NOTE: In the constructor we enforced that bkgErr can be 0 only when also bkgCounts = 0
        # Also it is evident from the expression above that when bkgCounts = 0 and bkgErr=0 also b=0

        # Let's do the branch with background > 0 first

        idx = self.bkgCounts > 0

        log_likes = np.empty_like(modelCounts)

        log_likes[idx] = ( -(b[idx] - self.bkgCounts[idx])**2 / (2 * s2[idx])
                           + self.counts[idx] * np.log(b[idx] + modelCounts[idx])
                           - b[idx] - modelCounts[idx] - logfactorial(self.counts[idx])
                           - 0.5 * log(2*np.pi) - np.log(self.bkgErr[idx]))

        # Let's do the other branch

        nidx = ~idx

        # the 1e-100 in the log is to avoid zero divisions
        # This is the Poisson likelihood with no background
        log_likes[nidx] = self.counts[nidx] * np.log(modelCounts[nidx] + 1e-100) - modelCounts[nidx] \
                          - logfactorial(self.counts[nidx])

        return np.sum(log_likes)

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


class Response(object):
    def __init__(self, rspfile):

        rspNumber = 1

        # Check if we are dealing with a .rsp2 file (containing more than
        # one response). This is checked by looking for the syntax
        # [responseFile]{[responseNumber]}

        if ('{' in rspfile):
            tokens = rspfile.split("{")
            rspfile = tokens[0]
            rspNumber = int(tokens[-1].split('}')[0].replace(" ", ""))

        # Read the response
        with pyfits.open(rspfile) as f:

            try:

                # GBM typical response
                data = f['MATRIX', rspNumber].data

            except:
                # Other detectors might use the SPECRESP MATRIX name instead
                # Note that here we are not catching any exception, because
                # we have to fail if we cannot read the matrix

                data = f['SPECRESP MATRIX', rspNumber].data

            # Sometimes .rsp files contains a weird format featuring variable-length
            # arrays. Historically those confuse pyfits quite a lot, so we ensure
            # to transform them into standard numpy matrices to avoid issues

            self.matrix = variableToMatrix(data.field('MATRIX'))

            self.ebounds = np.vstack([f['EBOUNDS'].data.field("E_MIN"),
                                         f['EBOUNDS'].data.field("E_MAX")]).T

            self.mc_channels = np.vstack([data.field("ENERG_LO"),
                                             data.field("ENERG_HI")]).T

    def setFunction(self, differentialFunction, integralFunction=None):
        '''
        Set the function to be used for the convolution
        '''

        self.differentialFunction = differentialFunction

        if (integralFunction == None):

            # This should never happen in 3ML, but we keep this functionality for
            # other uses. Print a warning anyway

            warnings.warn("Using the slow numerical integration in the GBM plugin!",
                          RuntimeWarning)

            def integral(x, y):
                return scipy.integrate.quad(self.differentialFunction, x, y)[0]

            # NB: vectorize is super slow!
            self.integralFunction = np.vectorize(integral, otypes=[np.float])

        else:

            self.integralFunction = integralFunction

    def convolve(self):

        trueFluxes = self.integralFunction(self.mc_channels[:, 0],
                                           self.mc_channels[:, 1])

        foldedCounts = np.dot(trueFluxes, self.matrix.T)

        return foldedCounts

    def getCountsVector(self, e1, e2):

        trueFluxes = self.integralFunction(self.mc_channels[:, 0], self.mc_channels[:, 1])

    def energyToChannel(self, energy):

        '''Finds the channel containing the provided energy.
        NOTE: returns the channel index (starting at zero),
        not the channel number (likely starting from 1)'''

        # Get the index of the first ebounds upper bound larger than energy

        try:

            idx = next(idx for idx,
                               value in enumerate(self.ebounds[:, 1])
                       if value >= energy)

        except StopIteration:

            # No values above the given energy, return the last channel
            return self.ebounds[:, 1].shape[0]

        return idx


def variableToMatrix(variableLengthMatrix):
    '''This take a variable length array and return it in a
    properly formed constant length array, to avoid some pyfits obscure bugs'''

    nrows = len(variableLengthMatrix)
    ncolumns = max([len(elem) for elem in variableLengthMatrix])
    matrix = np.zeros([ncolumns, nrows])

    for i in range(nrows):
        for j in range(ncolumns):

            try:

                matrix[j, i] = variableLengthMatrix[i][j]

            except:

                # This happens when the i,j element does not exist, which is not an error
                # We will just leave it to zero
                pass

    return matrix
