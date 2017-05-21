__author__ = 'grburgess'

import numpy as np

from threeML.io.file_utils import file_existing_and_readable
from threeML.plugins.OGIP.pha import PHAII
from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.plotting.light_curve_plots import binned_light_curve_plot
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.OGIP.pha import PHAWrite
from threeML.utils.stats_tools import Significance

from threeML.exceptions.custom_exceptions import deprecated

import copy

__instrument_name = "Generic EventList data"


class BinningMethodError(RuntimeError):
    pass


class EventListLike(OGIPLike):
    @deprecated('Please use the TimeSeriesBuilder for event list data')
    def __init__(self, name, event_list, rsp_file, source_intervals, background_selections=None,
                 poly_order=-1, unbinned=True, verbose=True, restore_poly_fit=None):
        """
        Generic EventListLike that should be inherited
        """




        assert (background_selections is not None) or (
            restore_poly_fit is not None), "you specify background selections or a restore file"

        self._event_list = event_list

        self._event_list.poly_order = poly_order

        # Fit the background and
        # Obtain the counts for the initial input interval
        # which is embedded in the background call

        self._startup = True  # This keeps things from being called twice!

        source_intervals = [interval.replace(' ', '') for interval in source_intervals.split(',')]

        self.set_active_time_interval(*source_intervals)

        if restore_poly_fit is None:

            background_selections = [interval.replace(' ', '') for interval in background_selections.split(',')]

            self.set_background_interval(*background_selections, unbinned=unbinned)

        else:

            if file_existing_and_readable(restore_poly_fit):

                self._event_list.restore_fit(restore_poly_fit)

                # In theory this will automatically get the poly counts if a
                # time interval already exists

                self._bkg_pha = PHAII.from_time_series(self._event_list, use_poly=True)



            else:

                if background_selections is None:

                    raise RuntimeError(
                        "Could not find saved background %s and no background_selections specified" % restore_poly_fit)

                else:

                    custom_warnings.warn(
                        "Could not find saved background %s. Fitting background manually" % restore_poly_fit)

                    background_selections = [interval.replace(' ', '') for interval in background_selections.split(',')]

                    self.set_background_interval(*background_selections, unbinned=unbinned)

        # Keeps track of if we are beginning
        self._startup = False

        # Keep track of if there has been any temporal binning

        # self._temporally_binned = False

        self._rsp_file = rsp_file

        self._verbose = verbose

        super(EventListLike, self).__init__(name,
                                            observation=self._observed_pha,
                                            background=self._bkg_pha,
                                            response=rsp_file,
                                            verbose=verbose,
                                            spectrum_number=1)

    @classmethod
    def _new_plugin(cls, *args, **kwargs):

        # because the inner class is actaully
        # OGIPLike, we need to explicitly call it here
        return OGIPLike._new_plugin(*args, **kwargs)

    def _output(self):

        super_out = super(EventListLike, self)._output()
        return super_out.append(self._event_list._output())

    def __set_poly_order(self, value):
        """Background poly order setter """

        self._event_list.poly_order = value

    def ___set_poly_order(self, value):
        """ Indirect poly order setter """

        self.__set_poly_order(value)

    def __get_poly_order(self):
        """ Get poly order """
        return self._event_list.poly_order

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

        self._event_list.set_active_time_intervals(*intervals)

        self._observed_pha = PHAII.from_time_series(self._event_list, use_poly=False)

        self._active_interval = intervals

        if not self._startup:
            self._bkg_pha = PHAII.from_time_series(self._event_list, use_poly=True)

            super(EventListLike, self).__init__(self.name,
                                                observation=self._observed_pha,
                                                background=self._bkg_pha,
                                                response=self._rsp_file,
                                                verbose=self._verbose,
                                                spectrum_number=1)

        self._tstart = self._event_list.time_intervals.absolute_start_time
        self._tstop = self._event_list.time_intervals.absolute_stop_time

        return_ogip = False

        if 'return_ogip' in kwargs:
            return_ogip = bool(kwargs.pop('return_ogip'))

        if return_ogip:
            # I really do not like this at the moment
            # but I'm assuming there is only one interval selected
            new_name = "%s_%s" % (self._name, intervals[0])

            new_ogip = OGIPLike(new_name,
                                observation=self._observed_pha,
                                background=self._bkg_pha,
                                response=self._rsp_file,
                                verbose=self._verbose,
                                spectrum_number=1)

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
        if 'unbinned' in options:

            unbinned = options.pop('unbinned')
        else:

            unbinned = self._default_unbinned

        self._event_list.set_polynomial_fit_interval(*intervals, unbinned=unbinned)

        # In theory this will automatically get the poly counts if a
        # time interval already exists

        self._bkg_pha = PHAII.from_time_series(self._event_list, use_poly=True)

        if not self._startup:
            super(EventListLike, self).__init__(self.name,
                                                observation=self._observed_pha,
                                                background=self._bkg_pha,
                                                response=self._rsp_file,
                                                verbose=self._verbose,
                                                spectrum_number=1)

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

        for interval in self._event_list.bins:
            self.set_active_time_interval(interval.to_string())

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

        return self._event_list.get_poly_info()

    def save_background(self, filename, overwrite=False):
        """

        save the background to and HDF5 file. The filename does not need an extension.
        The filename will be saved as <filename>_bkg.h5



        :param filename: name of file to save
        :param overwrite: to overwrite or not
        :return:
        """

        self._event_list.save_background(filename, overwrite)

    def view_lightcurve(self, start=-10, stop=20., dt=1., use_binner=False, energy_selection=None,
                        significance_level=None, instrument='n.a.'):
        # type: (float, float, float, bool, str, float, str) -> None

        """
        :param instrument:
        :param start:
        :param stop:
        :param dt:
        :param use_binner:
        :param energy_selection:
        :param significance_level:
        """

        if energy_selection is not None:

            # we can go through and filter out those channels that do not correspond to
            # out energy selection

            energy_selection = [interval.replace(' ', '') for interval in energy_selection.split(',')]

            valid_channels = []
            mask = np.array([False] * self._event_list.n_events)

            for selection in energy_selection:

                ee = list(map(float, selection.split("-")))

                if len(ee) != 2:
                    raise RuntimeError('Energy selection is not valid! Form: <low>-<high>.')

                emin, emax = sorted(ee)

                idx1 = self._rsp.energy_to_channel(emin)
                idx2 = self._rsp.energy_to_channel(emax)

                # Update the allowed channels
                valid_channels.extend(list(range(idx1, idx2)))

                this_mask = np.logical_and(self._event_list.energies >= idx1, self._event_list.energies <= idx2)

                np.logical_or(mask, this_mask, out=mask)

        else:

            mask = np.array([True] * self._event_list.n_events)
            valid_channels = list(range(self._event_list.n_channels))

        if use_binner:

            # we will use the binner object to bin the
            # light curve and ignore the normal linear binning

            bins = self._event_list.bins.time_edges

            # perhaps we want to look a little before or after the binner
            if start < bins[0]:
                pre_bins = np.arange(start, bins[0], dt).tolist()[:-1]

                pre_bins.extend(bins)

                bins = pre_bins

            if stop > bins[-1]:
                post_bins = np.arange(bins[-1], stop, dt)

                bins.extend(post_bins[1:])

        else:

            # otherwise, just use regular linear binning

            bins = np.arange(start, stop + dt, dt)

        cnts, bins = np.histogram(self._event_list.arrival_times[mask], bins=bins)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        width = np.diff(bins)

        # now we want to get the estimated background from the polynomial fit

        bkg = []
        for j, tb in enumerate(time_bins):
            tmpbkg = 0.
            for i in valid_channels:
                poly = self._event_list.polynomials[i]

                tmpbkg += poly.integral(tb[0], tb[1])

            bkg.append(tmpbkg / width[j])

        # here we will create a filter for the bins that exceed a certain
        # significance level

        if significance_level is not None:

            raise NotImplementedError("significnace filter is not complete")

            # create a significance object

            significance = Significance(Non=cnts / width, Noff=bkg)

            # we will go thru and get the background errors
            # for the current binned light curve

            bkg_err = []
            for j, tb in enumerate(time_bins):
                tmpbkg = 0.
                for i in valid_channels:
                    poly = self._event_list.polynomials[i]

                    tmpbkg += poly.integral_error(tb[0], tb[1]) ** 2

                bkg_err.append(np.sqrt(tmpbkg) / width[j])

            # collect the significances for this light curve and this binning

            lightcurve_sigma = significance.li_and_ma_equivalent_for_gaussian_background(sigma_b=np.asarray(bkg_err))

            print(lightcurve_sigma)

            # now create a filter for the bins that exceed the significance

            sig_filter = lightcurve_sigma >= significance_level

            if self._verbose:
                print(('time bins with significance greater that %f are shown in green' % significance_level))

        else:

            sig_filter = significance_level

        # pass all this to the light curve plotter

        binned_light_curve_plot(time_bins=time_bins,
                                cnts=cnts,
                                width=width,
                                bkg=bkg,
                                selection=self._event_list.time_intervals.bin_stack,
                                bkg_selections=self._event_list.poly_intervals.bin_stack
                                )

    @property
    def bins(self):

        return self._event_list.bins

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

            self._event_list.bin_by_constant(start, stop, dt)


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

            self._event_list.bin_by_significance(start, stop, sigma=sigma, min_counts=min_counts, mask=mask)


        elif method == 'bayesblocks':

            if 'p0' in options:

                p0 = options.pop('p0')

            else:

                p0 = 0.1

            if 'use_background' in options:

                use_background = options.pop('use_background')

            else:

                use_background = False

            self._event_list.bin_by_bayesian_blocks(start, stop, p0, use_background)

        elif method == 'custom':

            if type(start) is not list:

                if type(start) is not np.ndarray:
                    raise RuntimeError('start must be and array in custom mode')

            if type(stop) is not list:

                if type(stop) is not np.ndarray:
                    raise RuntimeError('stop must be and array in custom mode')

            assert len(start) == len(stop), 'must have equal number of start and stop times'

            self._event_list.bin_by_custom(start, stop)




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



        for i, interval in enumerate(self._event_list.bins):
            self.set_active_time_interval(interval.to_string())

            new_name = "%s_%d" % (self._name, i)

            new_ogip = OGIPLike(new_name,
                                observation=self._observed_pha,
                                background=self._bkg_pha,
                                response=self._rsp_file,
                                verbose=self._verbose,
                                spectrum_number=1)

            ogip_list.append(new_ogip)

        # restore the old interval

        self.set_active_time_interval(*old_interval)

        self._verbose = old_verbose

        return ogip_list
