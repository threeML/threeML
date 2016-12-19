__author__ = 'grburgess'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import re

try:

    import requests

except ImportError:

    has_requests = False

else:

    has_requests = True

import copy

from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.OGIP.eventlist import EventListWithDeadTime
from threeML.io.rich_display import display

from threeML.io.step_plot import step_plot
from threeML.plugins.OGIP.pha import PHAWrite

from threeML.config.config import threeML_config

__instrument_name = "Generic EventList data"


class BinningMethodError(RuntimeError):
    pass


class EventListLike(OGIPLike):
    def __init__(self, name, event_list, background_selections, source_intervals, rsp_file,
                 poly_order=-1, unbinned=True, verbose=True):
        """
        If the input files are TTE files. Background selections are specified as
        a comma separated string e.g. "-10-0,10-20"

        Initial source selection is input as a string e.g. "0-5"

        One can choose a background polynomial order by hand (up to 4th order)
        or leave it as the default polyorder=-1 to decide by LRT test

        FermiGBM_TTE_Like("GBM","glg_tte_n6_bn080916412.fit","-10-0,10-20","0-5","rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        """

        self._evt_list = event_list

        self._evt_list.poly_order = poly_order

        # Fit the background and
        # Obtain the counts for the initial input interval
        # which is embedded in the background call




        self._startup = True  # This keeps things from being called twice!

        source_intervals = [interval.replace(' ', '') for interval in source_intervals.split(',')]
        background_selections = [interval.replace(' ', '') for interval in background_selections.split(',')]

        self.set_active_time_interval(*source_intervals)
        self.set_background_interval(*background_selections, unbinned=unbinned)

        # Keeps track of if we are beginning
        self._startup = False

        # Keep track of if there has been any temporal binning

        # self._temporally_binned = False

        self._rsp_file = rsp_file

        self._verbose = verbose

        OGIPLike.__init__(self, name, pha_file=self._observed_pha, bak_file=self._bkg_pha, rsp_file=rsp_file,
                          verbose=verbose)

    def __set_poly_order(self, value):
        """Background poly order setter """

        self._evt_list.poly_order = value

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ Get poly order """
        return self._evt_list.poly_order

    def ___get_poly_order(self):
        """ Indirect poly order getter """

        return self.__get_poly_order()

    background_poly_order = property(___get_poly_order, ___set_poly_order,
                                     doc="Get or set the background polynomial order")

    def set_active_time_interval(self, *intervals, **kwargs):
        """
        Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        :param options:
        :param intervals:
        :return:
        """

        self._evt_list.set_active_time_intervals(*intervals)

        self._observed_pha = self._evt_list.get_pha_container(use_poly=False)

        self._active_interval = intervals

        if not self._startup:

            self._bkg_pha = self._evt_list.get_pha_container(use_poly=True)

            OGIPLike.__init__(self, self.name,
                              pha_file=self._observed_pha,
                              bak_file=self._bkg_pha,
                              rsp_file=self._rsp_file,
                              verbose=self._verbose)

        self._tstart = min(self._evt_list.tmin_list)
        self._tstop = max(self._evt_list.tmax_list)

        return_ogip = False

        if 'return_ogip' in kwargs:

            return_ogip = bool(kwargs.pop('return_ogip'))

        if return_ogip:

            # I really do not like this at the moment
            # but I'm assuming there is only one interval selected
            new_name = "%s_%s" % (self._name, intervals[0])

            new_ogip = OGIPLike(new_name,
                                pha_file=self._observed_pha,
                                bak_file=self._bkg_pha,
                                rsp_file=self._rsp_file,
                                verbose=self._verbose)

            return new_ogip

    def set_background_interval(self, *intervals, **options):
        """
        Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        setBackgroundInterval("-10.0-0.0","10.-15.")


        :param *intervals:
        :param **options:

        :return: none

        """

        self._evt_list.set_polynomial_fit_interval(*intervals, **options)

        # In theory this will automatically get the poly counts if a
        # time interval already exists

        self._bkg_pha = self._evt_list.get_pha_container(use_poly=True)

        if not self._startup:

            OGIPLike.__init__(self, self.name, pha_file=self._observed_pha, bak_file=self._bkg_pha,
                              rsp_file=self._rsp_file, verbose=self._verbose)

    def view_lightcurve(self, start=-10, stop=20., dt=1., use_binner=False, energy_selection=None):
        """

        :param use_binner: use the bins created via a binner
        :param start: start time to view
        :param stop:  stop time to view
        :param dt:  dt of the light curve
        :param energy_selection: string containing energy interval
        :return: fig
        """

        if energy_selection is not None:

            energy_selection = [interval.replace(' ', '') for interval in energy_selection.split(',')]

            valid_channels = []
            mask = np.array([False] * self._evt_list.n_events)

            for selection in energy_selection:

                ee = map(float, selection.split("-"))

                if len(ee) != 2:
                    raise RuntimeError('Energy selection is not valid! Form: <low>-<high>.')

                emin, emax = sorted(ee)

                idx1 = self._rsp.energy_to_channel(emin)
                idx2 = self._rsp.energy_to_channel(emax)

                # Update the allowed channels
                valid_channels.extend(range(idx1, idx2))

                this_mask = np.logical_and(self._evt_list.energies >= idx1, self._evt_list.energies <= idx2)

                np.logical_or(mask, this_mask, out=mask)

        else:

            mask = np.array([True] * self._evt_list.n_events)
            valid_channels = range(self._gbm_tte_file.n_channels)

        if use_binner:

            bin_start, bin_stop = self._evt_list.bins
            bins = bin_start.tolist() + [bin_stop.tolist()[-1]]  # type: list

            # perhaps we want to look a little before or after the binner
            if start < bins[0]:

                pre_bins = np.arange(start, bins[0], dt).tolist()[:-1]

                pre_bins.extend(bins)

                bins = pre_bins

            if stop > bins[-1]:

                post_bins = np.arange(bins[-1], stop, dt)

                bins.extend(post_bins[1:])

        else:

            bins = np.arange(start, stop + dt, dt)

        cnts, bins = np.histogram(self._gbm_tte_file.arrival_times[mask] - self._gbm_tte_file.triggertime, bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        width = np.diff(bins)

        bkg = []
        for j, tb in enumerate(time_bins):
            tmpbkg = 0.
            for i in valid_channels:
                poly = self._evt_list.polynomials[i]

                tmpbkg += poly.integral(tb[0], tb[1]) / (width[j])

            bkg.append(tmpbkg)

        gbm_light_curve_plot(time_bins, cnts, bkg, width,
                             selection=zip(self._evt_list.tmin_list, self._evt_list._tmax_list),
                             bkg_selections=self._evt_list.poly_intervals)

    def peek(self):

        print "TTE File Info:"

        self._evt_list.peek()

        print 'Timing Info:'

        self._gbm_tte_file.peek()

    def write_pha_from_binner(self, file_name, overwrite=False):
        """

        :param file_name:
        :param overwrite:
        :return:
        """

        # save the original interval if there is one
        old_interval = copy.copy(self._active_interval)
        old_verbose = copy.copy(self._verbose)

        self._verbose = False

        ogip_list = []

        # create copies of the OGIP plugins with the
        # time interval saved.

        for interval in self.text_bins:

            self.set_active_time_interval(interval)

            ogip_list.append(copy.copy(self))

        # write out the PHAII file

        pha_writer = PHAWrite(*ogip_list)

        pha_writer.write(file_name, overwrite=overwrite)

        # restore the old interval

        self.set_active_time_interval(*old_interval)

        self._verbose = old_verbose

    def get_background_parameters(self):
        """
        Returns a pandas DataFrame containing the background polynomial
        coefficients for each cahnnel.

        Returns:

            background dataframe

        """

        return self._evt_list.get_poly_info()

    @property
    def text_bins(self):

        return self._evt_list.text_bins

    @property
    def bins(self):

        return self._evt_list.bins

    def read_bins(self, ttelike):
        """

        Read the temporal bins from another *binned* FermiGBMTTELike instance
        and apply those bins to this instance

        :param ttelike: *binned* FermiGBMTTELike instance
        :return:
        """

        start, stop = ttelike.bins
        self.create_time_bins(start, stop, method='custom')

    def create_time_bins(self, start, stop, method='constant', **options):
        """

        Create time bins from start to stop with a given method (constant, siginificance, bayesblocks, custom).
        Each method has required keywords specified in the parameters. Once created, this can be used as
        a JointlikelihoodSet generator, or as input for viewing the light curve.

        :param start: start of the bins or array of start times for custom mode
        :param stop: stop of the bins or array of stop times for custom mode
        :param method: constant, significance, bayesblocks, custom
        :param use_energy_mask: (optional) use the energy mask when binning (default false)
        :param dt: <constant method> delta time of the
        :param sigma: <significance> sigma level of bins
        :param min_counts: (optional) <significance> minimum number of counts per bin
        :param p0: <bayesblocks> the chance probability of having the correct bin configuration.
        :return:
        """

        if 'use_energy_mask' in options:

            use_energy_mask = options.pop('use_energy_mask')

        else:

            use_energy_mask = False

        if method == 'constant':

            if 'dt' in options:
                dt = float(options.pop('dt'))

            else:

                raise RuntimeError('constant bins requires the dt option set!')

            self._evt_list.bin_by_constant(start, stop, dt)


        elif method == 'significance':

            if 'sigma' in options:

                sigma = options.pop('sigma')

            else:

                raise RuntimeError('significance bins require a sigma argument')

            if 'min_counts' in options:

                min_counts = options.pop('min_counts')

            else:

                min_counts = 10

            # should we mask the data

            if use_energy_mask:

                mask = self._mask

            else:

                mask = None

            self._evt_list.bin_by_significance(start, stop, sigma=sigma, min_counts=min_counts, mask=mask)


        elif method == 'bayesblocks':

            if 'p0' in options:

                p0 = options.pop('p0')

            else:

                p0 = 0.1

            if 'use_background' in options:

                use_background = options.pop('use_background')

            else:

                use_background = False

            self._evt_list.bin_by_bayesian_blocks(start, stop, p0, use_background)

        elif method == 'custom':

            if type(start) is not list:

                if type(start) is not np.ndarray:

                    raise RuntimeError('start must be and array in custom mode')

            if type(stop) is not list:

                if type(stop) is not np.ndarray:

                    raise RuntimeError('stop must be and array in custom mode')

            assert len(start) == len(stop), 'must have equal number of start and stop times'

            self._evt_list.bin_by_custom(start, stop)




        else:

            raise BinningMethodError('Only constant, significance, bayesblock, or custom method argument accepted.')

    def get_ogip_from_binner(self):
        """

        Returns a list of ogip_instances corresponding to the
        time intervals created by the binner.

        :return: list of ogip instances for each time interval
        """

        # save the original interval if there is one
        old_interval = copy.copy(self._active_interval)
        old_verbose = copy.copy(self._verbose)

        self._verbose = False

        ogip_list = []

        # create copies of the OGIP plugins with the
        # time interval saved.



        for i, interval in enumerate(self.text_bins):

            self.set_active_time_interval(interval)

            new_name = "%s_%d" % (self._name, i)

            new_ogip = OGIPLike(new_name,
                                pha_file=self._observed_pha,
                                bak_file=self._bkg_pha,
                                rsp_file=self._rsp_file,
                                verbose=self._verbose)

            ogip_list.append(new_ogip)

        # restore the old interval

        self.set_active_time_interval(*old_interval)

        self._verbose = old_verbose

        return ogip_list
