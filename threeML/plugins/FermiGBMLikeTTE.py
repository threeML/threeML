__author__ = 'drjfunk'
import math
import re
import warnings

import astropy.io.fits as pyfits
import numpy as np
import scipy.integrate

from threeML.plugins.ogip import gbm_light_curve_plot
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.ogip import OGIPPluginPGstat

__instrument_name = "Fermi GBM TTE (all detectors)"


class FermiGBMLikeTTE(OGIPPluginPGstat, PluginPrototype):

    def __init__(self, name, ttefile, bkgselections, srcinterval, rspfile, polyorder=-1):
        """
        If the input files are TTE files. Background selections are specified as
        a comma separated string e.g. "-10-0,10-20"

        Initial source selection is input as a string e.g. "0-5"

        One can choose a background polynomial order by hand (up to 4th order)
        or leave it as the default polyorder=-1 to decide by LRT test
        
        FermiGBM_TTE_Like("GBM","glg_tte_n6_bn080916412.fit","-10-0,10-20","0-5","rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        """

        self.name = name

        OGIPPluginPGstat.__init__(self, name, rspfile, ttefile=ttefile)

        self._polyorder = polyorder

        self.ttefile = GBMTTEFile(ttefile)

        self._backgroundexists = False
        self._energyselectionexists = False

        # Start with an empty mask (the user will overwrite it using the
        # setActiveMeasurement method)
        self.mask = np.asarray(np.ones(self.ttefile.nchans), np.bool)

        # Fit the background and
        # Obtain the counts for the initial input interval
        # which is embeded in the background call

        # First get the initial tmin and tmax

        self.tmin, self.tmax = self._parse_time_interval(srcinterval)

        self.set_background_interval(*bkgselections.split(','))

    @staticmethod
    def _parse_time_interval(time_interval):

        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def set_active_time_interval(self, arg):
        '''Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self.tmin, self.tmax = self._parse_time_interval(arg)





        # First build the mas for the events in time
        timemask = np.logical_and(self.ttefile.events - self.ttefile.triggertime >= self.tmin,
                                  self.ttefile.events - self.ttefile.triggertime <= self.tmax)
        tmpcounts = []  # Temporary list to hold the total counts per chan
        tmpbackground = []  # Temporary list to hold the background counts per chan
        tmpbackgrounderr = []  # Temporary list to hold the background counts per chan

        for chan in range(self.ttefile.nchans):
            channelmask = self.ttefile.pha == chan
            countsmask = np.logical_and(channelmask, timemask)
            totalcounts = len(self.ttefile.events[countsmask])

            # Now integrate the appropriate background polynomial
            backgroundcounts = self._polynomials[chan].integral(self.tmin, self.tmax)
            backgrounderror = self._polynomials[chan].integralError(self.tmin, self.tmax)

            tmpcounts.append(totalcounts)
            tmpbackground.append(backgroundcounts)
            tmpbackgrounderr.append(backgrounderror)

        counts = np.array(tmpcounts)
        bkgCounts = np.array(tmpbackground)
        bkgErr = np.array(tmpbackgrounderr)

        # Calculate the exposure using the GBM dead time (Meegan et al. 2009)
        totaldeadtime = self.ttefile.deadtime[timemask].sum()

        exposure = (self.tmax - self.tmin) - totaldeadtime

        # Run through the proper checks on counts
        self._initialSetup(self.mask, counts, bkgCounts, exposure, bkgErr)

    def set_background_interval(self, *time_intervals_spec):
        '''Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        setBackgroundInterval("-10.0-0.0","10.-15.")
        '''

        self._backgroundselections = []

        for time_interval_spec in time_intervals_spec:

            t1, t2 = self._parse_time_interval(time_interval_spec)

            self._backgroundselections.append((t1,t2))

        self._backgroundselections = np.array(self._backgroundselections)

        # Fit the background with the given intervals
        self._FitBackground()

        # Since changing the background will alter the counts
        # We need to recalculate the source interval
        self.set_active_time_interval("%.5f-%.5f" % (self.tmin, self.tmax))

    def view_lightcurve(self, start=-10, stop=20., dt=1.):

        binner = np.arange(start, stop + dt, dt)
        cnts, bins = np.histogram(self.ttefile.events - self.ttefile.triggertime, bins=binner)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        bkg = []
        for tb in time_bins:
            tmpbkg = 0.  # Maybe I can do this perenergy at some point
            for poly in self._polynomials:
                tmpbkg += poly.integral(tb[0], tb[1]) / (dt)

            bkg.append(tmpbkg)

        gbm_light_curve_plot(time_bins, cnts, bkg, dt, selection=[self.tmin, self.tmax])

    def _fitGlobalAndDetermineOptimumGrade(self, cnts, bins):
        # Fit the sum of all the channels to determine the optimal polynomial
        # grade
        Nintervals = len(bins)

        # y                         = []
        # for i in range(Nintervals):
        #  y.append(np.sum(counts[i]))
        # pass
        # y                         = np.array(y)

        # exposure                  = np.array(data.field("EXPOSURE"))

        print("\nLooking for optimal polynomial grade:")

        # Fit all the polynomials

        minGrade = 0
        maxGrade = 4
        logLikelihoods = []

        for grade in range(minGrade, maxGrade + 1):

            polynomial, logLike = self._polyfit(bins, cnts, grade)

            logLikelihoods.append(logLike)

        # Found the best one
        deltaLoglike = np.array(map(lambda x: 2 * (x[0] - x[1]), zip(logLikelihoods[:-1], logLikelihoods[1:])))

        print("\ndelta log-likelihoods:")

        for i in range(maxGrade):

            print("%s -> %s: delta Log-likelihood = %s" % (i, i + 1, deltaLoglike[i]))


        print("")

        deltaThreshold = 9.0

        mask = (deltaLoglike >= deltaThreshold)

        if (len(mask.nonzero()[0]) == 0):

            # best grade is zero!
            bestGrade = 0

        else:

            bestGrade = mask.nonzero()[0][-1] + 1

        return bestGrade

    def _FitBackground(self):

        self._backgroundexists = True
        ## Seperate everything by energy channel

        # Select all the events that are in the background regions
        # and make a mask

        allbkgmasks = []

        for bkgsel in self._backgroundselections:

            allbkgmasks.append(np.logical_and(self.ttefile.events - self.ttefile.triggertime >= bkgsel[0],
                                              self.ttefile.events - self.ttefile.triggertime <= bkgsel[1]))
        backgroundmask = allbkgmasks[0]

        # If there are multiple masks:
        if len(allbkgmasks) > 1:
            for mask in allbkgmasks[1:]:
                backgroundmask = np.logical_or(backgroundmask, mask)

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to 1 sec intervals
        if self._polyorder == -1:
            totalbkgevents = self.ttefile.events[backgroundmask]
            binwidth = 1.
            cnts, bins = np.histogram(totalbkgevents - self.ttefile.triggertime,
                                      bins=np.arange(self.ttefile.startevents - self.ttefile.triggertime,
                                                     self.ttefile.stopevents - self.ttefile.triggertime,
                                                     binwidth))

            cnts = cnts/binwidth
            # Find the mean time of the bins
            meantime = []
            for i in xrange(len(bins) - 1):
                m = np.mean((bins[i], bins[i + 1]))
                meantime.append(m)
            meantime = np.array(meantime)

            # Remove bins with zero counts
            allnonzeromask = []
            for bkgsel in self._backgroundselections:
                allnonzeromask.append(np.logical_and(meantime >= bkgsel[0],
                                                     meantime <= bkgsel[1]))

            nonzeromask = allnonzeromask[0]
            if len(allnonzeromask) > 1:
                for mask in allnonzeromask[1:]:
                    nonzeromask = np.logical_or(mask, nonzeromask)

            self.optimalPolGrade = self._fitGlobalAndDetermineOptimumGrade(cnts[nonzeromask], meantime[nonzeromask])

        else:

            self.optimalPolGrade = self._polyorder

        polynomials = []

        for chan in range(self.ttefile.nchans):

            # index all events for this channel and select them
            channelmask = self.ttefile.pha == chan
            # channelselectedevents = self.ttefile.events[channelmask]
            # Mask background events and current channel
            bkgchanmask = np.logical_and(backgroundmask, channelmask)
            # Select the masked events
            currentevents = self.ttefile.events[bkgchanmask]

            binwidth = 1.
            cnts, bins = np.histogram(currentevents - self.ttefile.triggertime,
                                      bins=np.arange(self.ttefile.startevents - self.ttefile.triggertime,
                                                     self.ttefile.stopevents - self.ttefile.triggertime,
                                                     binwidth))

            cnts = cnts / binwidth

            # Find the mean time of the bins
            meantime = []
            for i in xrange(len(bins) - 1):
                m = np.mean((bins[i], bins[i + 1]))
                meantime.append(m)
            meantime = np.array(meantime)

            # Remove bins with zero counts
            allnonzeromask = []
            for bkgsel in self._backgroundselections:
                allnonzeromask.append(np.logical_and(meantime >= bkgsel[0],
                                                     meantime <= bkgsel[1]))

            nonzeromask = allnonzeromask[0]
            if len(allnonzeromask) > 1:
                for mask in allnonzeromask[1:]:
                    nonzeromask = np.logical_or(mask, nonzeromask)

            # Finally, we fit the background and add the polynomial to a list
            thispolynomial, cstat = self._fitChannel(cnts[nonzeromask], meantime[nonzeromask], self.optimalPolGrade)
            polynomials.append(thispolynomial)

        self._polynomials = polynomials

    def _fitChannel(self, cnts, bins, polGrade):

        Nintervals = len(bins)
        # Put data to fit in an x vector and y vector

        polynomial, minLogLike = self._polyfit(bins, cnts, polGrade)

        return polynomial, minLogLike

    def _polyfit(self, x, y, polGrade):

        test = False

        # Check that we have enough counts to perform the fit, otherwise
        # return a "zero polynomial"
        nonzeroMask = (y > 0)
        Nnonzero = len(nonzeroMask.nonzero()[0])
        if (Nnonzero == 0):
            # No data, nothing to do!
            return Polynomial([0.0]), 0.0
        pass

        # Compute an initial guess for the polynomial parameters,
        # with a least-square fit (with weight=1) using SVD (extremely robust):
        # (note that polyfit returns the coefficient starting from the maximum grade,
        # thus we need to reverse the order)
        if (test):
            print("  Initial estimate with SVD..."),
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            initialGuess = np.polyfit(x, y, polGrade)
        pass
        initialGuess = initialGuess[::-1]
        if (test):
            print("  done -> %s" % (initialGuess))

        polynomial = Polynomial(initialGuess)

        # Check that the solution found is meaningful (i.e., definite positive
        # in the interval of interest)
        M = polynomial(x)
        negativeMask = (M < 0)
        if (len(negativeMask.nonzero()[0]) > 0):
            # Least square fit failed to converge to a meaningful solution
            # Reset the initialGuess to reasonable value
            initialGuess[0] = np.mean(y)
            meanx = np.mean(x)
            initialGuess = map(lambda x: abs(x[1]) / pow(meanx, x[0]), enumerate(initialGuess))

        # Improve the solution using a logLikelihood statistic (Cash statistic)
        logLikelihood = BkgLogLikelihood(x, y, polynomial)

        # Check that we have enough non-empty bins to fit this grade of polynomial,
        # otherwise lower the grade
        dof = Nnonzero - (polGrade + 1)
        if (test):
            print("Effective dof: %s" % (dof))
        if (dof <= 2):
            # Fit is poorly or ill-conditioned, have to reduce the number of parameters
            while (dof < 2 and len(initialGuess) > 1):
                initialGuess = initialGuess[:-1]
                polynomial = Polynomial(initialGuess)
                logLikelihood = BkgLogLikelihood(x, y, polynomial)
            pass
        pass

        # Try to improve the fit with the log-likelihood
        # try:
        if (1 == 1):
            finalEstimate = scipy.optimize.fmin(logLikelihood, initialGuess,
                                                ftol=1E-5, xtol=1E-5,
                                                maxiter=1e6, maxfun=1E6,
                                                disp=False)
        # except:
        else:
            # We shouldn't get here!
            raise RuntimeError("Fit failed! Try to reduce the degree of the polynomial.")
        pass

        # Get the value for cstat at the minimum
        minlogLikelihood = logLikelihood(finalEstimate)

        # Update the polynomial with the fitted parameters,
        # and the relative covariance matrix
        finalPolynomial = Polynomial(finalEstimate)
        try:
            finalPolynomial.computeCovarianceMatrix(logLikelihood.getFreeDerivs)
        except Exception:
            raise
        # if test is defined, compare the results with those obtained with ROOT


        return finalPolynomial, minlogLikelihood


class GBMTTEFile(object):
    def __init__(self, ttefile):
        '''
        A simple class for opening and easily accessing Fermi GBM
        TTE Files.

        :param ttefile: The filename of the TTE file to be stored

        '''

        tte = pyfits.open(ttefile)

        self.events = tte['EVENTS'].data['TIME']
        self.pha = tte['EVENTS'].data['PHA']
        self.triggertime = tte['PRIMARY'].header['TRIGTIME']
        self.startevents = tte['PRIMARY'].header['TSTART']
        self.stopevents = tte['PRIMARY'].header['TSTOP']
        self.nchans = tte['EBOUNDS'].header['NAXIS2']

        self._calculate_deattime()


    def _calculate_deattime(self):

        self.deadtime = np.zeros_like(self.events)
        overflowmask =  self.pha == 128

        # Dead time for overflow (note, overflow sometimes changes)
        self.deadtime[overflowmask] = 10.E-6 #s

        # Normal dead time
        self.deadtime[~overflowmask] = 2.E-6 #s


class BkgLogLikelihood(object):
    '''
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    '''

    def __init__(self, x, y, model, **kwargs):
        self.x = x
        self.y = y
        self.model = model
        self.parameters = model.getParams()

        # Initialize the exposure to 1.0 (i.e., non-influential)
        # It will be replaced by the real exposure if the exposure keyword
        # have been used
        self.exposure = np.zeros(len(x)) + 1.0

        for key in kwargs.keys():
            if (key.lower() == "exposure"):
                self.exposure = np.array(kwargs[key])
        pass

    pass

    def _evalLogM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        nontinyMask = (M > 2.0 * tiny)
        tinyMask = np.logical_not(nontinyMask)

        if (len(tinyMask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tinyMask] = np.abs(M[tinyMask]) / tiny + np.log(tiny) - 1
            logM[nontinyMask] = np.log(M[nontinyMask])
        else:
            logM = np.log(M)
        return logM

    pass

    def __call__(self, parameters):
        '''
          Evaluate the Cash statistic for the given set of parameters
        '''

        # Compute the values for the model given this set of parameters
        self.model.setParams(parameters)
        M = self.model(self.x) * self.exposure
        Mfixed, tiny = self._fixPrecision(M)

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero
        negativeMask = (M < 0)
        if (len(negativeMask.nonzero()[0]) > 0):
            M[negativeMask] = 0.0
        pass

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evalLogM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0
        d_times_logM = np.zeros(len(self.y))
        nonzeroMask = (self.y > 0)
        d_times_logM[nonzeroMask] = self.y[nonzeroMask] * logM[nonzeroMask]

        logLikelihood = np.sum(Mfixed - d_times_logM)

        return logLikelihood

    pass

    def _fixPrecision(self, v):
        '''
          Round extremely small number inside v to the smallest usable
          number of the type corresponding to v. This is to avoid warnings
          and errors like underflows or overflows in math operations.
        '''
        tiny = np.float64(np.finfo(v[0]).tiny)
        zeroMask = (np.abs(v) <= tiny)
        if (len(zeroMask.nonzero()[0]) > 0):
            v[zeroMask] = np.sign(v[zeroMask]) * tiny

        return v, tiny

    pass

    def getFreeDerivs(self, parameters=None):
        '''
        Return the gradient of the logLikelihood for a given set of parameters (or the current
        defined one, if parameters=None)
        '''
        # The derivative of the logLikelihood statistic respect to parameter p is:
        # dC / dp = Sum [ (dM/dp)_i - D_i/M_i (dM/dp)_i]

        # Get the number of parameters and initialize the gradient to 0
        Nfree = self.model.getNumFreeParams()
        derivs = np.zeros(Nfree)

        # Set the parameters, if a new set has been provided
        if (parameters != None):
            self.model.setParams(parameters)
        pass

        # Get the gradient of the model respect to the parameters
        modelDerivs = self.model.getFreeDerivs(self.x) * self.exposure
        # Get the model
        M = self.model(self.x) * self.exposure

        M, tinyM = self._fixPrecision(M)

        # Compute y_divided_M = y/M: inizialize y_divided_M to zero
        # and then overwrite the elements for which y > 0. This is to avoid
        # possible underflow and overflow due to the finite precision of the
        # computer
        y_divided_M = np.zeros(len(self.y))
        nonzero = (self.y > 0)
        y_divided_M[nonzero] = self.y[nonzero] / M[nonzero]

        for p in range(Nfree):
            thisModelDerivs, tinyMd = self._fixPrecision(modelDerivs[p])
            derivs[p] = np.sum(thisModelDerivs * (1.0 - y_divided_M))
        pass

        return derivs

    pass


pass


class Polynomial(object):
    def __init__(self, params):
        self.params = params
        self.degree = len(params) - 1

        # Build an empty covariance matrix
        self.covMatrix = np.zeros([self.degree + 1, self.degree + 1])

    pass

    def horner(self, x):
        """A function that implements the Horner Scheme for evaluating a
        polynomial of coefficients *args in x."""
        result = 0
        for coefficient in self.params[::-1]:
            result = result * x + coefficient
        return result

    pass

    def __call__(self, x):
        return self.horner(x)

    pass

    def __str__(self):
        # This is call by the print() command
        # Print results
        output = "\n------------------------------------------------------------"
        output += '\n| {0:^10} | {1:^20} | {2:^20} |'.format("COEFF", "VALUE", "ERROR")
        output += "\n|-----------------------------------------------------------"
        for i, parValue in enumerate(self.getParams()):
            output += '\n| {0:<10d} | {1:20.5g} | {2:20.5g} |'.format(i, parValue, math.sqrt(self.covMatrix[i, i]))
        pass
        output += "\n------------------------------------------------------------"

        return output

    pass

    def setParams(self, parameters):
        self.params = parameters

    pass

    def getParams(self):
        return self.params

    pass

    def getNumFreeParams(self):
        return self.degree + 1

    pass

    def getFreeDerivs(self, x):
        Npar = self.degree + 1
        freeDerivs = []
        for i in range(Npar):
            freeDerivs.append(map(lambda xx: pow(xx, i), x))
        pass
        return np.array(freeDerivs)

    pass

    def computeCovarianceMatrix(self, statisticGradient):
        self.covMatrix = computeCovarianceMatrix(statisticGradient, self.params)
        # Check that the covariance matrix is positive-defined
        negativeElements = (np.matrix.diagonal(self.covMatrix) < 0)
        if (len(negativeElements.nonzero()[0]) > 0):
            raise RuntimeError(
                "Negative element in the diagonal of the covariance matrix. Try to reduce the polynomial grade.")

    pass

    def getCovarianceMatrix(self):
        return self.covMatrix

    pass

    def integral(self, xmin, xmax):
        '''
        Evaluate the integral of the polynomial between xmin and xmax
        '''
        integralCoeff = [0]
        integralCoeff.extend(map(lambda i: self.params[i - 1] / float(i), range(1, self.degree + 1 + 1)))

        integralPolynomial = Polynomial(integralCoeff)

        return integralPolynomial(xmax) - integralPolynomial(xmin)

    pass

    def integralError(self, xmin, xmax):
        # Based on http://root.cern.ch/root/html/tutorials/fit/ErrorIntegral.C.html

        # Set the weights
        i_plus_1 = np.array(range(1, self.degree + 1 + 1), 'd')

        def evalBasis(x):
            return (1 / i_plus_1) * pow(x, i_plus_1)

        c = evalBasis(xmax) - evalBasis(xmin)

        # Compute the error on the integral
        err2 = 0.0
        nPar = self.degree + 1
        parCov = self.getCovarianceMatrix()
        for i in range(nPar):
            s = 0.0
            for j in range(nPar):
                s += parCov[i, j] * c[j]
            pass
            err2 += c[i] * s
        pass

        return math.sqrt(err2)

    pass


pass


def computeCovarianceMatrix(grad, par, full_output=False,
                            init_step=0.01, min_step=1e-12, max_step=1, max_iters=50,
                            target=0.1, min_func=1e-7, max_func=4):
    """Perform finite differences on the _analytic_ gradient provided by user to calculate hessian/covariance matrix.

    Positional args:
        grad                : a function to return a gradient
        par                 : vector of parameters (should be function minimum for covariance matrix calculation)

    Keyword args:

        full_output [False] : if True, return information about convergence, else just the covariance matrix
        init_step   [1e-3]  : initial step size (0.04 ~ 10% in log10 space); can be a scalar or vector
        min_step    [1e-6]  : the minimum step size to take in parameter space
        max_step    [1]     : the maximum step size to take in parameter sapce
        max_iters   [5]     : maximum number of iterations to attempt to converge on a good step size
        target      [0.5]   : the target change in the function value for step size
        min_func    [1e-4]  : the minimum allowable change in (abs) function value to accept for convergence
        max_func    [4]     : the maximum allowable change in (abs) function value to accept for convergence
    """

    nparams = len(par)
    step_size = np.ones(nparams) * init_step
    step_size = np.maximum(step_size, min_step * 1.1)
    step_size = np.minimum(step_size, max_step * 0.9)
    hess = np.zeros([nparams, nparams])
    min_flags = np.asarray([False] * nparams)
    max_flags = np.asarray([False] * nparams)

    def revised_step(delta_f, current_step, index):
        if (current_step == max_step):
            max_flags[i] = True
            return True, 0

        elif (current_step == min_step):
            min_flags[i] = True
            return True, 0

        else:
            adf = abs(delta_f)
            if adf < 1e-8:
                # need to address a step size that results in a likelihood change that's too
                # small compared to precision
                pass

            if (adf < min_func) or (adf > max_func):
                new_step = current_step / (adf / target)
                new_step = min(new_step, max_step)
                new_step = max(new_step, min_step)
                return False, new_step
            else:
                return True, 0

    iters = np.zeros(nparams)
    for i in xrange(nparams):
        converged = False

        for j in xrange(max_iters):
            iters[i] += 1

            di = step_size[i]
            par[i] += di
            g_up = grad(par)

            par[i] -= 2 * di
            g_dn = grad(par)

            par[i] += di

            delta_f = (g_up - g_dn)[i]

            converged, new_step = revised_step(delta_f, di, i)
            # print 'Parameter %d -- Iteration %d -- Step size: %.2e -- delta: %.2e'%(i,j,di,delta_f)

            if converged:
                break
            else:
                step_size[i] = new_step
        pass

        hess[i, :] = (g_up - g_dn) / (2 * di)  # central difference

        if not converged:
            print 'Warning: step size for parameter %d (%.2g) did not result in convergence.' % (i, di)
    try:
        cov = np.linalg.inv(hess)
    except:
        print 'Error inverting hessian.'
        raise Exception('Error inverting hessian')
    if full_output:
        return cov, hess, step_size, iters, min_flags, max_flags
    else:
        return cov
    pass


pass
