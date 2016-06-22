# Implements a minimal reader for OGIP PHA format for spectral data
# (https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node6.html)

# Author: Giacomo Vianello (giacomov@slac.stanford.edu)

import os
import warnings
from math import log

import matplotlib.pyplot as plt

from threeML.io.step_plot import step_plot
from threeML.plugins.gammaln import logfactorial


import astropy.io.fits as pyfits
import numpy as np
from astromodels.parameter import Parameter

from threeML.io import file_utils

requiredKeywords = {}
requiredKeywords['observed'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER," +
                                "exposure:EXPOSURE,backfile:BACKFILE," +
                                "corrfile:CORRFILE,corrscal:CORRSCAL,respfile:RESPFILE," +
                                "ancrfile:ANCRFILE,hduclass:HDUCLASS," +
                                "hduclas1:HDUCLAS1,hduvers:HDUVERS,poisserr:POISSERR," +
                                "chantype:CHANTYPE,n_channels:DETCHANS").split(",")
requiredKeywords['background'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER," +
                                  "exposure:EXPOSURE," +
                                  "hduclass:HDUCLASS," +
                                  "hduclas1:HDUCLAS1,hduvers:HDUVERS,poisserr:POISSERR," +
                                  "chantype:CHANTYPE,n_channels:DETCHANS").split(",")
mightBeColumns = {}
mightBeColumns['observed'] = ("EXPOSURE,BACKFILE," +
                              "CORRFILE,CORRSCAL," +
                              "RESPFILE,ANCRFILE").split(",")
mightBeColumns['background'] = ("EXPOSURE").split(",")


class OGIPPluginBase(object):

    def __init__(self, name, response, **other_datafiles):

        self.name = name

        # Check that all file exists

        all_files = list(other_datafiles.values())
        all_files.append(response)

        for filename in all_files:

            if not file_utils.file_existing_and_readable(filename.split("{")[0]):
                raise IOError("File %s does not exists!" % filename)

        self.response = Response(response, other_datafiles.get("arffile"))

        self.counts = None
        self.bkgCounts = None
        self.mask = None
        self.exposure = None

    def _initialSetup(self, mask, counts, bkgCounts, exposure, bkgErr):

        self.mask = mask
        self.counts = counts
        self.bkgCounts = bkgCounts
        self.exposure = exposure
        self.bkgErr = bkgErr

        # Check that counts is positive
        idx = (self.counts < 0)

        if (np.sum(idx) > 0):
            warnings.warn("The observed spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.counts[idx] = 0

        pass

        # Get the background counts for this spectrum
        # self.bkgCounts = (self.bkgfile.getRates()[self.mask]
        #                  * self.exposure)

        # Check that bkgCounts is positive

        idx = (self.bkgCounts < 0)

        if (np.sum(idx) > 0):
            warnings.warn("The background spectrum for %s " % self.name +
                          "has negative channels! Fixing those to zero.",
                          RuntimeWarning)
            self.bkgCounts[idx] = 0

        # Check that errors are zeros when the bkg is zero
        if (bkgErr is not None) and np.any((self.bkgErr == 0) & (self.bkgCounts > 0)):
            raise RuntimeError("The background error is zero but the background counts are not zero for some bins. "
                               "Data are corrupted.")

        # Check that the observed counts are positive

        idx = self.counts < 0

        if np.sum(idx) > 0:
            raise RuntimeError("Negative counts in observed spectrum %s. Data are corrupted." % ('fix this'))

        # Keep a copy which will never be modified
        self.counts_backup = np.array(self.counts, copy=True)
        self.bkgCounts_backup = np.array(self.bkgCounts, copy=True)

        if bkgErr is not None:
            self.bkgErr_backup = np.array(self.bkgErr, copy=True)

        self.counts = self.counts[self.mask]
        self.bkgCounts = self.bkgCounts[self.mask]

        if bkgErr is not None:

            self.bkgErr = self.bkgErr[self.mask]

            # Check that errors are zeros when the bkg is zero
            if np.any((self.bkgErr == 0) & (self.bkgCounts > 0)):
                raise RuntimeError("The background error is zero but the background counts are not zero for some bins. "
                                   "Data are corrupted.")

        # Effective area correction is disabled by default, i.e.,
        # the nuisance parameter is fixed to 1
        self.nuisanceParameters = {}
        self.nuisanceParameters['InterCalib'] = Parameter("InterCalib", 1, min_value=0.9, max_value=1.1, delta=0.01)
        self.nuisanceParameters['InterCalib'].fix = True

    def set_model(self, likelihoodModel):
        '''
        Set the model to be used in the joint minimization.
        '''
        self.likelihoodModel = likelihoodModel

        diffFlux, integral = self._get_diff_flux_and_integral()

        # This is a wrapper which iterates over all the point sources and get
        # the fluxes
        # We assume there are no extended sources, since we cannot handle them here

        assert self.likelihoodModel.get_number_of_extended_sources() == 0, "OGIP-like plugins do not support " \
                                                                           "extended sources"



        self.response.set_function(diffFlux, integral)

    def _get_diff_flux_and_integral(self):

        nPointSources = self.likelihoodModel.get_number_of_point_sources()

        def diffFlux(energies):
            fluxes = self.likelihoodModel.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, nPointSources):
                fluxes += self.likelihoodModel.get_point_source_fluxes(i, energies)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):

            # Simpson's rule

            return (e2 - e1) / 6.0 * (diffFlux(e1)
                                      + 4 * diffFlux((e1 + e2) / 2.0)
                                      + diffFlux(e2))

        return diffFlux, integral

    def get_name(self):
        '''
        Return a name for this dataset (likely set during the constructor)
        '''
        return self.name

    def inner_fit(self):

        # ad the moment there is no fitting here. We might introduce the renormalization constant
        # at some point

        return self.get_log_like()

    def get_folded_model(self):

        # Get the folded model for this spectrum
        # (this is the rate predicted, in cts/s)

        return self.response.convolve()[self.mask]

    def _get_model_counts(self):

        # Get the folded model for this spectrum (this is the rate predicted,
        # in cts/s)

        folded = self.get_folded_model()

        # Model is folded+background (i.e., we assume negligible errors on the
        # background)
        modelCounts = self.nuisanceParameters['InterCalib'].value * folded * self.exposure

        # Put all negative predicted counts to zero
        clippedModelCounts = np.clip(modelCounts, 0, 1e10)

        return clippedModelCounts

    def set_active_measurements(self, *args):
        '''Set the measurements to be used during the analysis.
        Use as many ranges as you need,
        specified as 'emin-emax'. Energies are in keV. Example:

        setActiveMeasurements('10-12.5','56.0-100.0')

        which will set the energy range 10-12.5 keV and 56-100 keV to be
        used in the analysis'''

        # To implelemnt this we will use an array of boolean index,
        # which will filter
        # out the non-used channels during the logLike

        # Now build the new mask: values for which the mask is 0 will be masked
        mask = np.zeros_like(self.counts_backup)

        for arg in args:

            ee = map(float, arg.replace(" ", "").split("-"))
            emin, emax = sorted(ee)

            idx1 = self.response.energy_to_channel(emin)
            idx2 = self.response.energy_to_channel(emax)

            mask[idx1:idx2 + 1] = True

        self.mask = np.array(mask, np.bool)

        self.counts = self.counts_backup[self.mask]
        self.bkgCounts = self.bkgCounts_backup[self.mask]

        if self.bkgErr is not None:

            self.bkgErr = self.bkgErr_backup[self.mask]

        print("Now using %s channels out of %s" % (np.sum(self.mask),
                                                   self.mask.shape[0]
                                                   ))

    def _compute_log_like(self, modelCounts):

        raise NotImplementedError("You have to override this with the proper statistic")

    def get_log_like(self):
        '''
        Return the value of the log-likelihood with the current values for the
        parameters
        '''

        modelCounts = self._get_model_counts()

        logLike = self._compute_log_like(modelCounts)

        return logLike

    def get_nuisance_parameters(self):
        '''
        Return a list of nuisance parameter names. Return an empty list if there
        are no nuisance parameters
        '''
        return self.nuisanceParameters.keys()

    def view_count_spectrum(self):
        '''
        View the count and background spectrum. Useful to check energy selections.

        '''
        # First plot the counts
        _ = gbm_channel_plot(self.response.ebounds[:, 0], self.response.ebounds[:, 1], self.counts_backup,
                             color='#377eb8', lw=2, alpha=1)
        ax = gbm_channel_plot(self.response.ebounds[:, 0], self.response.ebounds[:, 1], self.bkgCounts_backup,
                              color='#e41a1c', alpha=.8)
        # Now fade the non-used channels
        excluded_channel_plot(self.response.ebounds[:, 0], self.response.ebounds[:, 1], self.mask, self.counts_backup,
                              self.bkgCounts_backup, ax)

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts/keV")
        ax.set_xlim(left=self.response.ebounds[0, 0], right=self.response.ebounds[-1, 1])

    def use_intercalibration_constant(self, factorLowBound=0.9, factorHiBound=1.1):

        self.nuisanceParameters['InterCalib'].free()
        self.nuisanceParameters['InterCalib'].set_bounds(factorLowBound, factorHiBound)

        # Check that the parameter is within the provided bounds
        value = self.nuisanceParameters['InterCalib'].value

        if (value < factorLowBound):
            warnings.warn(
                "The intercalibration constant was %s, lower than the provided lower bound %s." % (
                    value, factorLowBound) +
                " Setting it equal to the lower bound")

            self.nuisanceParameters['InterCalib'].setValue(float(factorLowBound))

        if (value > factorHiBound):
            warnings.warn(
                "The intercalibration constant was %s, larger than the provided hi bound %s." % (value, factorHiBound) +
                " Setting it equal to the hi bound")

            self.nuisanceParameters['InterCalib'].setValue(float(factorHiBound))

    def fix_intercalibration_constant(self, value=None):

        if (value is not None):
            # Fixing the constant to the provided value
            self.nuisanceParameters['InterCalib'].value = float(value)

        else:

            # Do nothing, i.e., leave the constant to the value
            # it currently has
            pass

        self.nuisanceParameters['InterCalib'].fix()

    def _chisq(self, modelCounts):

        # for chisq we need sigma^2, but sigma = sqrt(counts), -> sigma^2 = counts
        # If there are zero counts then the denominator is 0, so use the model
        # variance instead

        sanitized_variance = np.where(self.counts > 0, self.counts, modelCounts)
        sanitized_variance = np.maximum(sanitized_variance, 1e-12)

        # This not really chisq, but it's a Gaussian likelihood instead. That's why there is a -1/2
        # in front of it. However, this like = -0.5 * chisq + const (we neglect the const)

        log_likes = - 0.5 * (modelCounts - self.counts) ** 2 / sanitized_variance

        return np.sum(log_likes)

    def use_chisq(self):

        self._backup_log_like = self._compute_log_like
        self._compute_log_like = self._chisq

    def restore_normal_statistic(self):

        self._compute_log_like = self._backup_log_like


class OGIPPluginPGstat(OGIPPluginBase):

    # Override the _initialSetup method just to force the existance of bkgErr
    def _initialSetup(self, mask, counts, bkgCounts, exposure, bkgErr):

        assert bkgErr is not None, "You have to provide errors on the background to use PGstat"

        super(OGIPPluginPGstat, self)._initialSetup(mask, counts, bkgCounts, exposure, bkgErr)

    def _compute_log_like(self, modelCounts):
        # This loglike assume Gaussian errors on the background and Poisson uncertainties on the
        # observed counts. It is a profile likelihood.

        MB = self.bkgCounts + modelCounts
        s2 = self.bkgErr ** 2

        b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * self.counts) + self.bkgErr ** 4)
                   + self.bkgCounts - modelCounts - s2)

        # Now there are two branches: when the background is 0 we are in the normal situation of a pure
        # Poisson likelihood, while when the background is not zero we use the profile likelihood

        # NOTE: In the constructor we enforced that bkgErr can be 0 only when also bkgCounts = 0
        # Also it is evident from the expression above that when bkgCounts = 0 and bkgErr=0 also b=0

        # Let's do the branch with background > 0 first

        idx = self.bkgCounts > 0

        log_likes = np.empty_like(modelCounts)

        log_likes[idx] = (-(b[idx] - self.bkgCounts[idx]) ** 2 / (2 * s2[idx])
                          + self.counts[idx] * np.log(b[idx] + modelCounts[idx])
                          - b[idx] - modelCounts[idx] - logfactorial(self.counts[idx])
                          - 0.5 * log(2 * np.pi) - np.log(self.bkgErr[idx]))

        # Let's do the other branch

        nidx = ~idx

        # the 1e-100 in the log is to avoid zero divisions
        # This is the Poisson likelihood with no background
        log_likes[nidx] = self.counts[nidx] * np.log(modelCounts[nidx] + 1e-100) - modelCounts[nidx] \
                          - logfactorial(self.counts[nidx])

        return np.sum(log_likes)


class OGIPPluginCash(OGIPPluginBase):

    # Override the _initialSetup method just to force the non-existance of bkgErr
    def _initialSetup(self, mask, counts, bkgCounts, exposure, bkgErr=None):

        assert bkgErr is None, "You cannot use bkg. errors when using Cash statistic"

        super(OGIPPluginCash, self)._initialSetup(mask, counts, bkgCounts, exposure, None)

    def _compute_log_like(self, modelCounts):

        log_likes = self.counts * np.log(modelCounts + 1e-100) - modelCounts - logfactorial(self.counts)

        return np.sum(log_likes)


class OGIPPHA(object):

    def __init__(self, phafile, spectrumNumber=None, **kwargs):

        if '.root' not in phafile:

            self._init_from_FITS(phafile, spectrumNumber, **kwargs)

        else:

            self._init_from_ROOT()

    def _init_from_ROOT(self):

        raise NotImplementedError("Not yet implemented")

    def _init_from_FITS(self, phafile, spectrumNumber, **kwargs):

        self.filetype = 'observed'

        for k, v in kwargs.iteritems():

            if (k.lower() == "filetype"):

                if (v.lower() == "background"):

                    self.filetype = "background"

                elif (v.lower() == "observed"):

                    self.filetype = "observed"

                else:

                    raise RuntimeError("Unrecognized filetype keyword value")

        # Allow the use of a syntax like "mySpectrum.pha{1}" to specify the spectrum
        # number in PHA II files

        ext = os.path.splitext(phafile)[-1]

        if ('{' in ext):
            spectrumNumber = int(ext.split('{')[-1].replace('}', ''))

            phafile = phafile.split('{')[0]

        with pyfits.open(phafile) as f:

            try:

                HDUidx = f.index_of("SPECTRUM")

            except:

                raise RuntimeError("The input file %s is not in PHA format" % (phafile))

            self.spectrumNumber = spectrumNumber

            spectrum = f[HDUidx]
            data = spectrum.data
            header = spectrum.header

            # Determine if this file contains COUNTS or RATES

            self.exposure = 1

            if ("COUNTS" in data.columns.names):

                self.hasRates = False
                self.dataColumnName = "COUNTS"

            elif ("RATE" in data.columns.names):

                self.hasRates = True
                self.dataColumnName = "RATE"

                #self.exposure = header.get("EXPOSURE")

            else:

                raise RuntimeError("This file does not contain a RATE nor a COUNTS column. "
                                   "This is not a valid PHA file")

            # Determine if this is a PHA I or PHA II
            if (len(data.field(self.dataColumnName).shape) == 2):

                self.typeII = True

                if (self.spectrumNumber == None):
                    raise RuntimeError("This is a PHA Type II file. You have to provide a spectrum number")

            else:

                self.typeII = False

            # Collect informations from mandatory keywords

            keys = requiredKeywords[self.filetype]

            for k in keys:

                internalName, keyname = k.split(":")

                if (keyname in header):

                    self.__setattr__(internalName, header[keyname])

                else:

                    if (keyname in mightBeColumns[self.filetype] and self.typeII):

                        # Check if there is a column with this name

                        if (keyname in data.columns.names):

                            # This will set the exposure, among other things

                            self.__setattr__(internalName, data[keyname][self.spectrumNumber - 1])

                        else:

                            raise RuntimeError("Keyword %s is not in the header nor in the data extension. "
                                               "This file is not a proper PHA file" % keyname)

                    else:

                        # The keyword POISSERR is a special case, because even if it is missing,
                        # it is assumed to be False if there is a STAT_ERR column in the file

                        if (keyname == "POISSERR" and "STAT_ERR" in data.columns.names):

                            self.poisserr = False

                        else:

                            raise RuntimeError("Keyword %s not found. File %s is not a proper PHA "
                                               "file" % (keyname, phafile))

            # Now get the data (counts or rates) and their errors. If counts, transform them in rates
            if (self.typeII):

                # PHA II file
                if (self.hasRates):

                    self.rates = data.field(self.dataColumnName)[self.spectrumNumber - 1, :]

                    if (not self.poisserr):
                        self.ratesErrors = data.field("STAT_ERR")[self.spectrumNumber - 1, :]

                else:

                    self.rates = data.field(self.dataColumnName)[self.spectrumNumber - 1, :] / self.exposure

                    if (not self.poisserr):

                        self.ratesErrors = data.field("STAT_ERR")[self.spectrumNumber - 1, :] / self.exposure


                if ("SYS_ERR" in data.columns.names):

                    self.sysErrors = data.field("SYS_ERR")[self.spectrumNumber - 1, :]
                else:

                    self.sysErrors = np.zeros(self.rates.shape)


            elif (self.typeII == False):

                # PHA 1 file
                if (self.hasRates):

                    self.rates = data.field(self.dataColumnName)

                    if (not self.poisserr):
                        self.ratesErrors = data.field("STAT_ERR")
                else:

                    self.rates = data.field(self.dataColumnName) / self.exposure

                    if (not self.poisserr):
                        self.ratesErrors = data.field("STAT_ERR") / self.exposure

                if ("SYS_ERR" in data.columns.names):

                    self.sysErrors = data.field("SYS_ERR")

                else:

                    self.sysErrors = np.zeros(self.rates.shape)

    def getRates(self):
        return self.rates

    def getRatesErrors(self):

        assert self.poisserr == False, "Cannot request errors on rates for a Poisson spectrum"

        return self.ratesErrors

    def getSysErrors(self):
        return self.sysErrors

    def getExposure(self):
        return self.exposure


class Response(object):
    def __init__(self, rspfile, arffile=None):

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

            # Now let's see if we have a ARF, if yes, read it

            if arffile is not None:

                with pyfits.open(arffile) as f:

                    data = f['SPECRESP'].data

                arf = data.field('SPECRESP')

                # Check that arf and rmf have same dimensions

                if arf.shape[0] != self.matrix.shape[1]:
                    raise IOError("The ARF and the RMF file does not have the same number of channels")

                # Check that the ENERG_LO and ENERG_HI for the RMF and the ARF
                # are the same

                arf_mc_channels = np.vstack([data.field("ENERG_LO"),
                                             data.field("ENERG_HI")]).T

                # Declare the mc channels different if they differ by more than
                # 1%

                idx = (self.mc_channels > 0)

                diff = (self.mc_channels[idx] - arf_mc_channels[idx]) / self.mc_channels[idx]

                if diff.max() > 0.01:
                    raise IOError("The ARF and the RMF have one or more MC channels which differ by more than 1%")

                # Multiply ARF and RMF

                self.matrix = self.matrix * arf

    def set_function(self, differentialFunction, integralFunction=None):
        '''
        Set the function to be used for the convolution
        '''

        self.differentialFunction = differentialFunction

        self.integralFunction = integralFunction

    def convolve(self):

        trueFluxes = self.integralFunction(self.mc_channels[:, 0],
                                           self.mc_channels[:, 1])

        # Sometimes some channels have 0 lenths, or maybe they start at 0, where
        # many functions (like a power law) are not defined. In the response these
        # channels have usually a 0, but unfortunately for a computer
        # inf * zero != zero. Thus, let's force this. We avoid checking this situation
        # in details because this would have a HUGE hit on performances

        idx = np.isfinite(trueFluxes)
        trueFluxes[~idx] = 0

        foldedCounts = np.dot(trueFluxes, self.matrix.T)

        return foldedCounts

    def getCountsVector(self, e1, e2):

        trueFluxes = self.integralFunction(self.mc_channels[:, 0], self.mc_channels[:, 1])

        return trueFluxes

    def energy_to_channel(self, energy):

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


def gbm_channel_plot(chan_min, chan_max, counts, **keywords):
    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min
    fig = plt.figure(666)
    ax = fig.add_subplot(111)
    step_plot(chans, counts / width, ax, **keywords)
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def excluded_channel_plot(chan_min, chan_max, mask, counts, bkg, ax):
    # Figure out the best limit
    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min

    top = max([max(bkg / width), max(counts / width)])
    top = top + top * .5
    bottom = min([min(bkg / width), min(counts / width)])
    bottom = bottom - bottom * .2

    # Find the contiguous regions
    slices = slice_disjoint((~mask).nonzero()[0])

    for region in slices:
        ax.fill_between([chan_min[region[0]], chan_max[region[1]]],
                        bottom,
                        top,
                        color='k',
                        alpha=.5)

    ax.set_ylim(bottom, top)


def slice_disjoint(arr):
    slices = []
    startSlice = 0
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            endSlice = arr[i]
            slices.append([startSlice, endSlice])
            startSlice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if endSlice != arr[-1]:
        slices.append([startSlice, arr[-1]])
    return slices


def gbm_light_curve_plot(time_bins, cnts, bkg, width, selection):
    fig = plt.figure(777)
    ax = fig.add_subplot(111)

    maxCnts = max(cnts / width)
    top = maxCnts + maxCnts * .2
    minCnts = min(cnts[cnts > 0] / width)
    bottom = minCnts - minCnts * .2
    mean_time = map(np.mean, time_bins)

    step_plot(time_bins, cnts / width, ax, color='#8da0cb')

    ax.plot(mean_time, bkg, '#66c2a5', lw=2.)

    ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)