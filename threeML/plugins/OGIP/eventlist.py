# Creates a generic event list reader that can create PHA objects on the fly

import numpy as np
import scipy
import re
import warnings
import math
import copy

import pandas as pd

from threeML.io.rich_display import display
from threeML.utils.stats_tools import li_and_ma
from pha import PHAContainer

# testing
import multiprocessing
import joblib


class EventList(object):
    def __init__(self, arrival_times, energies, n_channels, start_time=None, stop_time=None, dead_time=None,
                 first_channel=0, rsp_file=None, ra=None, dec=None):
        """
        Container for event style data which are tagged with time and energy/PHA.


        Args:
            n_channels: Number of detector channels
            start_time: start time of the event list
            stop_time: stop time of the event list
            dead_time: an array of deadtime per event
            first_channel: where detchans begin indexing
            rsp_file: the response file corresponding to these events
            arrival_times: list of event arrival times
            energies: list of event energies or pha channels
            ra:
            dec:
        """

        self._arrival_times = np.asarray(arrival_times)
        self._energies = np.asarray(energies)
        self._n_channels = n_channels
        self._first_channel = first_channel

        assert self._arrival_times.shape[0] == self._energies.shape[
            0], "Arrival time (%d) and energies (%d) have different shapes" % (
            self._arrival_times.shape[0], self._energies.shape[0])

        if dead_time is not None:

            self._dead_time = np.asarray(dead_time)

            assert self._arrival_times.shape[0] == self._dead_time.shape[
                0], "Arrival time (%d) and Dead Time (%d) have different shapes" % (
                self._arrival_times.shape[0], self._dead_time.shape[0])

        else:

            self._dead_time = None

        if start_time is None:

            self._start_time = arrival_times.min()

        else:

            self._start_time = start_time

        if stop_time is None:

            self._stop_time = arrival_times.max()

        else:

            self._stop_time = stop_time

        self._rsp_file = rsp_file

        self._user_poly_order = -1
        self._time_selection_exists = False
        self._poly_fit_exists = False

    @staticmethod
    def _parse_time_interval(time_interval):
        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def set_active_time_intervals(self, *args):
        '''Set the time interval(s) to be used during the analysis.

        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_intervals("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self._time_selection_exists = True

        # try:
        #     use_poly_fit = kwargs.pop('use_poly_fit')
        # except(KeyError):
        #     use_poly_fit = False

        tmin_list = []
        tmax_list = []
        interval_masks = []

        for arg in args:
            tmin, tmax = self._parse_time_interval(arg)
            mask = np.logical_and(self._arrival_times >= tmin,
                                  self._arrival_times <= tmax)

            tmin_list.append(tmin)
            tmax_list.append(tmax)
            interval_masks.append(mask)

        if intervals_overlap(tmin_list, tmax_list):
            raise RuntimeError('Provided intervals are overlapping and hence invalid')

        time_mask = interval_masks[0]
        if len(interval_masks) > 1:
            for mask in interval_masks[1:]:
                time_mask = np.logical_or(time_mask, mask)

        tmp_counts = []  # Temporary list to hold the total counts per chan

        for chan in range(self._first_channel, self._n_channels + self._first_channel):
            channel_mask = self._energies == chan
            counts_mask = np.logical_and(channel_mask, time_mask)
            total_counts = len(self._arrival_times[counts_mask])

            tmp_counts.append(total_counts)

        self._counts = np.array(tmp_counts)

        # self._is_poisson = True


        tmp_counts = []
        tmp_err = []  # Temporary list to hold the err counts per chan

        if self._poly_fit_exists:

            if not self._poly_fit_exists:
                raise RuntimeError('A polynomial fit to the channels does not exist!')

            for chan in range(self._first_channel, self._n_channels + self._first_channel):

                total_counts = 0
                counts_err = 0

                for tmin, tmax in zip(tmin_list, tmax_list):
                    # Now integrate the appropriate background polynomial
                    total_counts += self._polynomials[chan].integral(tmin, tmax)
                    counts_err += (self._polynomials[chan].integral_error(tmin, tmax)) ** 2

                tmp_counts.append(total_counts)

                tmp_err.append(np.sqrt(counts_err))

            self._poly_counts = np.array(tmp_counts)

            self._poly_count_err = np.array(tmp_err)

            # self._is_poisson = False

        # Dead time correction

        exposure = 0.
        for tmin, tmax in zip(tmin_list, tmax_list):
            exposure += tmax - tmin

        if self._dead_time is not None:

            total_dead_time = self._dead_time[time_mask].sum()
        else:

            total_dead_time = 0.

        self._exposure = exposure - total_dead_time
        self._total_dead_time = total_dead_time

        self._tmin_list = tmin_list
        self._tmax_list = tmax_list

        self._active_dead_time = total_dead_time

    @property
    def tmin_list(self):
        return self._tmin_list

    @property
    def tmax_list(self):
        return self._tmax_list

    @property
    def polynomials(self):
        """ Returns polynomial is they exist"""
        if self._poly_fit_exists:
            return self._polynomials
        else:
            RuntimeError('A polynomial fit has not been made.')

    def __set_poly_order(self, value):
        """ Set poly order only in allowed range and redo fit """

        assert type(value) is int, "Polynomial order must be integer"

        assert -1 <= value <= 4, "Polynomial order must be 0-4 or -1 to have it determined"

        self._user_poly_order = value

        if self._poly_fit_exists:

            print('Refitting background with new polynomial order and existing selections')
            self._fit_background()

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ get the poly order """

        return self._user_poly_order

    def ___get_poly_order(self):
        """ Indirect poly order getter """

        return self.__get_poly_order()

    poly_order = property(___get_poly_order, ___set_poly_order,
                          doc="Get or set the polynomial order")

    def set_polynomial_fit_interval(self, *time_intervals_spec):
        """Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_polynomial_fit_interval("-10.0-0.0","10.-15.")

        Args:
            *time_intervals_spec:
        """

        self._poly_time_selections = []

        for time_interval in time_intervals_spec:
            t1, t2 = self._parse_time_interval(time_interval)

            self._poly_time_selections.append((t1, t2))

        self._poly_time_selections = np.array(self._poly_time_selections)

        # Fit the events with the given intervals
        self._fit_background()

        # Since changing the poly fit will alter the counts
        # We need to recalculate the source interval

        self._poly_fit_exists = True
        if self._time_selection_exists:

            tmp = []
            for tmin, tmax in zip(self._tmin_list, self._tmax_list):
                tmp.append("%.5f-%.5f" % (tmin, tmax))

            self.set_active_time_intervals(*tmp)

    def get_pha_container(self, use_poly=False):
        """
        Return a PHAContainer that can be read by the PHA class


        Args:
            use_poly: (bool) choose to build from the polynomial fits

        Returns:

        """
        if not self._time_selection_exists:
            raise RuntimeError('No time selection exists! Cannot calculate rates')

        if use_poly:
            is_poisson = False

            rate_err = self._poly_count_err / self._exposure
            rates = self._poly_counts / self._exposure





        else:

            is_poisson = True

            rate_err = None
            rates = self._counts / (self._exposure)

        pha = PHAContainer(rates=rates,
                           rate_errors=rate_err,
                           n_channels=self._n_channels,
                           exposure=self._exposure,
                           is_poisson=is_poisson,
                           response_file=self._rsp_file
                           )

        return pha

    def peek(self):
        """
        Examine the currently selected info as well other things.

        """

        info_dict = {}

        info_dict['Active Selections'] = zip(self._tmin_list, self._tmax_list)
        info_dict['Active Deadtime'] = self._active_dead_time
        info_dict['Active Exposure'] = self._exposure
        info_dict['Total N. Events'] = len(self._arrival_times)
        info_dict['Active Counts'] = self._counts.sum()
        info_dict['Number of Channels'] = self._n_channels

        if self._poly_fit_exists:
            info_dict['Polynomial Selections'] = self._poly_time_selections
            info_dict['Polynomial Order'] = self._optimal_polynomial_grade
            info_dict['Active Count Error'] = np.sqrt((self._poly_count_err ** 2).sum())
            info_dict['Active Polynomial Counts'] = self._poly_counts.sum()

            S = li_and_ma(self._counts.sum(), self._poly_counts.sum())

            info_dict['Li and Ma Sigma'] = S  # not sure if li and ma applies here

        info_df = pd.Series(info_dict)

        display(info_df)

    def _fit_global_and_determine_optimum_grade(self, cnts, bins):
        # Fit the sum of all the channels to determine the optimal polynomial
        # grade
        Nintervals = len(bins)

        # y                         = []
        # for i in range(Nintervals):
        #  y.append(np.sum(counts[i]))
        # pass
        # y                         = np.array(y)

        # exposure                  = np.array(data.field("EXPOSURE"))

        # print("\nLooking for optimal polynomial grade:")

        # Fit all the polynomials

        min_grade = 0
        max_grade = 4
        log_likelihoods = []

        for grade in range(min_grade, max_grade + 1):
            polynomial, log_like = _polyfit(bins, cnts, grade)

            log_likelihoods.append(log_like)

        # Found the best one
        deltaLoglike = np.array(map(lambda x: 2 * (x[0] - x[1]), zip(log_likelihoods[:-1], log_likelihoods[1:])))

        # print("\ndelta log-likelihoods:")

        # for i in range(max_grade):
        #    print("%s -> %s: delta Log-likelihood = %s" % (i, i + 1, deltaLoglike[i]))

        # print("")

        deltaThreshold = 9.0

        mask = (deltaLoglike >= deltaThreshold)

        if (len(mask.nonzero()[0]) == 0):

            # best grade is zero!
            bestGrade = 0

        else:

            bestGrade = mask.nonzero()[0][-1] + 1

        return bestGrade

    def _fit_background(self):

        self._poly_fit_exists = True
        ## Separate everything by energy channel

        # Select all the events that are in the background regions
        # and make a mask

        all_bkg_masks = []

        for bkgsel in self._poly_time_selections:
            all_bkg_masks.append(np.logical_and(self._arrival_times >= bkgsel[0],
                                                self._arrival_times <= bkgsel[1]))
        poly_mask = all_bkg_masks[0]

        # If there are multiple masks:
        if len(all_bkg_masks) > 1:
            for mask in all_bkg_masks[1:]:
                poly_mask = np.logical_or(poly_mask, mask)

        # Now we will find the the best poly order unless the use specified one
        # The total cnts (over channels) is binned to 1 sec intervals
        if self._user_poly_order == -1:
            totalbkgevents = self._arrival_times[poly_mask]
            bin_width = .1
            cnts, bins = np.histogram(totalbkgevents,
                                      bins=np.arange(self._start_time,
                                                     self._stop_time,
                                                     bin_width))

            cnts = cnts / bin_width
            # Find the mean time of the bins
            mean_time = []
            for i in xrange(len(bins) - 1):
                m = np.mean((bins[i], bins[i + 1]))
                mean_time.append(m)
            mean_time = np.array(mean_time)

            # Remove bins with zero counts
            all_non_zero_mask = []
            
            for bkgsel in self._poly_time_selections:
                all_non_zero_mask.append(np.logical_and(mean_time >= bkgsel[0],
                                                        mean_time <= bkgsel[1]))

            non_zero_mask = all_non_zero_mask[0]
            if len(all_non_zero_mask) > 1:
                for mask in all_non_zero_mask[1:]:
                    non_zero_mask = np.logical_or(mask, non_zero_mask)

            self._optimal_polynomial_grade = self._fit_global_and_determine_optimum_grade(cnts[non_zero_mask],
                                                                                          mean_time[non_zero_mask])

            print("Auto-determined polynomial order: %d" % self._optimal_polynomial_grade)


        else:

            self._optimal_polynomial_grade = self._user_poly_order

        # Attempting a parallel execution

        channels = range(self._first_channel, self._n_channels + self._first_channel)
        num_cpus = multiprocessing.cpu_count()
        polynomials = joblib.Parallel(n_jobs=num_cpus)(
                joblib.delayed(_fit_channel)(chan,
                                             poly_mask,
                                             self._start_time,
                                             self._stop_time,
                                             self._energies,
                                             self._arrival_times,
                                             self._poly_time_selections,
                                             self._optimal_polynomial_grade) for chan in channels)

        # polynomials = []
        #
        # for chan in range(self._first_channel, self._n_channels + self._first_channel):
        #
        #     this_polynomial, cstat = self._fit_channel(chan, poly_mask)
        #
        #     polynomials.append(this_polynomial)


        self._polynomials = polynomials


def _fit_channel(channel, poly_mask, start_time, stop_time, energies, arrival_times, poly_selections, grade):
    """ Fit each channel of the data. Function to allow parallel execution

    Args:
        channel:
        poly_mask:
        start_time:
        stop_time:
        energies:
        arrival_times:
        poly_selections:
        grade:
    """
    # index all events for this channel and select them
    channel_mask = energies == channel

    # Mask background events and current channel
    poly_chan_mask = np.logical_and(poly_mask, channel_mask)
    # Select the masked events
    current_events = arrival_times[poly_chan_mask]

    bin_width = .1

    cnts, bins = np.histogram(current_events,
                              bins=np.arange(start_time,
                                             stop_time,
                                             bin_width))

    cnts = cnts / bin_width

    # Find the mean time of the bins
    mean_time = []
    for i in xrange(len(bins) - 1):
        m = np.mean((bins[i], bins[i + 1]))
        mean_time.append(m)
    mean_time = np.array(mean_time)

    # Remove bins with zero counts
    all_non_zero_mask = []
    for bkgsel in poly_selections:
        all_non_zero_mask.append(np.logical_and(mean_time >= bkgsel[0],

                                                mean_time <= bkgsel[1]))

    non_zero_mask = all_non_zero_mask[0]
    if len(all_non_zero_mask) > 1:
        for mask in all_non_zero_mask[1:]:
            non_zero_mask = np.logical_or(mask, non_zero_mask)

    # Put data to fit in an x vector and y vector
    polynomial, min_log_like = _polyfit(mean_time[non_zero_mask],
                                        cnts[non_zero_mask],
                                        grade)

    return polynomial


def _polyfit(x, y, grade):
    """ funtion to fit a polynomial to event data. not a member to allow parallel computation """
    test = False

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    non_zero_mask = (y > 0)
    n_non_zero = len(non_zero_mask.nonzero()[0])
    if n_non_zero == 0:
        # No data, nothing to do!
        return Polynomial([0.0]), 0.0
    pass

    # Compute an initial guess for the polynomial parameters,
    # with a least-square fit (with weight=1) using SVD (extremely robust):
    # (note that polyfit returns the coefficient starting from the maximum grade,
    # thus we need to reverse the order)
    if test:
        print("  Initial estimate with SVD..."),
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        initial_guess = np.polyfit(x, y, grade)
    pass
    initial_guess = initial_guess[::-1]
    if (test):
        print("  done -> %s" % (initial_guess))

    polynomial = Polynomial(initial_guess)

    # Check that the solution found is meaningful (i.e., definite positive
    # in the interval of interest)
    M = polynomial(x)

    negative_mask = (M < 0)

    if len(negative_mask.nonzero()[0]) > 0:
        # Least square fit failed to converge to a meaningful solution
        # Reset the initialGuess to reasonable value
        initial_guess[0] = np.mean(y)
        meanx = np.mean(x)
        initial_guess = map(lambda x: abs(x[1]) / pow(meanx, x[0]), enumerate(initial_guess))

    # Improve the solution using a logLikelihood statistic (Cash statistic)
    log_likelihood = PolyLogLikelihood(x, y, polynomial)

    # Check that we have enough non-empty bins to fit this grade of polynomial,
    # otherwise lower the grade
    dof = n_non_zero - (grade + 1)
    if test:
        print("Effective dof: %s" % (dof))
    if dof <= 2:
        # Fit is poorly or ill-conditioned, have to reduce the number of parameters
        while (dof < 2 and len(initial_guess) > 1):
            initial_guess = initial_guess[:-1]
            polynomial = Polynomial(initial_guess)
            log_likelihood = PolyLogLikelihood(x, y, polynomial)
        pass
    pass

    # Try to improve the fit with the log-likelihood
    # try:
    if (1 == 1):
        final_estimate = scipy.optimize.fmin(log_likelihood, initial_guess,
                                             ftol=1E-5, xtol=1E-5,
                                             maxiter=1e6, maxfun=1E6,
                                             disp=False)
    # except:
    else:
        # We shouldn't get here!
        raise RuntimeError("Fit failed! Try to reduce the degree of the polynomial.")
    pass

    # Get the value for cstat at the minimum
    min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix
    final_polynomial = Polynomial(final_estimate)

    try:
        final_polynomial.compute_covariance_matrix(log_likelihood.get_free_derivs)
    except Exception:
        raise
    # if test is defined, compare the results with those obtained with ROOT


    return final_polynomial, min_log_likelihood


class PolyLogLikelihood(object):
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
        self.parameters = model.get_params()

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
        self.model.set_params(parameters)
        M = self.model(self.x) * self.exposure
        Mfixed, tiny = self._fix_precision(M)

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

    def _fix_precision(self, v):
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

    def get_free_derivs(self, parameters=None):
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
        if (parameters is not None):
            self.model.set_params(parameters)
        pass

        # Get the gradient of the model respect to the parameters
        modelDerivs = self.model.get_free_derivs(self.x) * self.exposure
        # Get the model
        M = self.model(self.x) * self.exposure

        M, tinyM = self._fix_precision(M)

        # Compute y_divided_M = y/M: inizialize y_divided_M to zero
        # and then overwrite the elements for which y > 0. This is to avoid
        # possible underflow and overflow due to the finite precision of the
        # computer
        y_divided_M = np.zeros(len(self.y))
        nonzero = (self.y > 0)
        y_divided_M[nonzero] = self.y[nonzero] / M[nonzero]

        for p in range(Nfree):
            thisModelDerivs, tinyMd = self._fix_precision(modelDerivs[p])
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
        self.cov_matrix = np.zeros([self.degree + 1, self.degree + 1])

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
        for i, parValue in enumerate(self.get_params()):
            output += '\n| {0:<10d} | {1:20.5g} | {2:20.5g} |'.format(i, parValue, math.sqrt(self.cov_matrix[i, i]))
        pass
        output += "\n------------------------------------------------------------"

        return output

    pass

    def set_params(self, parameters):
        self.params = parameters

    pass

    def get_params(self):
        return self.params

    pass

    def getNumFreeParams(self):
        return self.degree + 1

    pass

    def get_free_derivs(self, x):
        Npar = self.degree + 1
        freeDerivs = []
        for i in range(Npar):
            freeDerivs.append(map(lambda xx: pow(xx, i), x))
        pass
        return np.array(freeDerivs)

    pass

    def compute_covariance_matrix(self, statisticGradient):
        self.cov_matrix = compute_covariance_matrix(statisticGradient, self.params)
        # Check that the covariance matrix is positive-defined
        negativeElements = (np.matrix.diagonal(self.cov_matrix) < 0)
        if (len(negativeElements.nonzero()[0]) > 0):
            raise RuntimeError(
                    "Negative element in the diagonal of the covariance matrix. Try to reduce the polynomial grade.")

    pass

    def get_covariance_matrix(self):
        return self.cov_matrix

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

    def integral_error(self, xmin, xmax):
        # Based on http://root.cern.ch/root/html/tutorials/fit/ErrorIntegral.C.html

        # Set the weights
        i_plus_1 = np.array(range(1, self.degree + 1 + 1), 'd')

        def evalBasis(x):
            return (1 / i_plus_1) * pow(x, i_plus_1)

        c = evalBasis(xmax) - evalBasis(xmin)

        # Compute the error on the integral
        err2 = 0.0
        nPar = self.degree + 1
        parCov = self.get_covariance_matrix()
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


def compute_covariance_matrix(grad, par, full_output=False,
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


def intervals_overlap(tmin, tmax):
    n_intervals = len(tmin)

    # Check that
    for i in range(n_intervals):
        throw_away_tmin = copy.copy(tmin)
        throw_away_tmax = copy.copy(tmax)

        this_min = throw_away_tmin.pop(i)
        this_max = throw_away_tmax.pop(i)

        for mn, mx in zip(throw_away_tmin, throw_away_tmax):

            if this_min < mn < this_max:

                return True

            elif this_min < mx < this_max:

                return True

        return False
