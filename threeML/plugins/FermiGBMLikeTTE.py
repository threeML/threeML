__author__ = 'drjfunk'

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

from OGIPLike import OGIPLike
from threeML.plugin_prototype import PluginPrototype
from OGIP.eventlist import EventList

from threeML.io.step_plot import step_plot

__instrument_name = "Fermi GBM TTE (all detectors)"


class FermiGBMLikeTTE(OGIPLike, PluginPrototype):
    def __init__(self, name, tte_file, background_selections, source_intervals, rsp_file, poly_order=-1):
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

        self._polyorder = poly_order

        self.ttefile = GBMTTEFile(tte_file)

        self._evt_list = EventList(arrival_times=self.ttefile.events - self.ttefile.triggertime,
                                   energies=self.ttefile.pha,
                                   n_channels=self.ttefile.nchans,
                                   start_time=self.ttefile.startevents - self.ttefile.triggertime,
                                   stop_time=self.ttefile.stopevents - self.ttefile.triggertime,
                                   dead_time=self.ttefile.deadtime,
                                   first_channel=0)

        self._evt_list.poly_order = self._polyorder

        self._backgroundexists = False
        self._energyselectionexists = False

        # Start with an empty mask (the user will overwrite it using the
        # setActiveMeasurement method)
        self.mask = np.asarray(np.ones(self.ttefile.nchans), np.bool)

        # Fit the background and
        # Obtain the counts for the initial input interval
        # which is embeded in the background call

        # First get the initial tmin and tmax

        # self.tmin, self.tmax = self._parse_time_interval(srcinterval)

        self.set_active_time_interval(*source_intervals.split(','))
        self.set_background_interval(*background_selections.split(','))

        OGIPLike.__init__(self, name, pha_file=self._observed_pha, bak_file=self._bkg_pha, rsp_file=rsp_file)

    def set_active_time_interval(self, *args):
        '''Set the time interval to be used during the analysis.
        For now, only one interval can be selected. This may be
        updated in the future to allow for self consistent time
        resolved analysis.
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        set_active_time_interval("0.0-10.0")

        which will set the energy range 0-10. seconds.
        '''

        self._evt_list.set_active_time_intervals(*args)

        self._observed_pha = self._evt_list.get_pha_container(use_poly=False)

        self._active_interval = args

    def set_background_interval(self, *time_intervals_spec):
        '''Set the time interval to fit the background.
        Multiple intervals can be input as separate arguments
        Specified as 'tmin-tmax'. Intervals are in seconds. Example:

        setBackgroundInterval("-10.0-0.0","10.-15.")
        '''

        self._evt_list.set_polynomial_fit_interval(*time_intervals_spec)

        # In theory this will automatically get the poly counts if a
        # time interval already exists

        self._bkg_pha = self._evt_list.get_pha_container(use_poly=True)

    def view_lightcurve(self, start=-10, stop=20., dt=1.):

        binner = np.arange(start, stop + dt, dt)
        cnts, bins = np.histogram(self.ttefile.events - self.ttefile.triggertime, bins=binner)
        time_bins = np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])

        bkg = []
        for tb in time_bins:
            tmpbkg = 0.  # Maybe I can do this perenergy at some point
            for poly in self._evt_list.polynomials:
                tmpbkg += poly.integral(tb[0], tb[1]) / (dt)

            bkg.append(tmpbkg)

        gbm_light_curve_plot(time_bins, cnts, bkg, dt,
                             selection=zip(self._evt_list.tmin_list, self._evt_list._tmax_list))

    def peek(self):

        self._evt_list.peek()


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
        overflowmask = self.pha == 128

        # Dead time for overflow (note, overflow sometimes changes)
        self.deadtime[overflowmask] = 10.E-6  # s

        # Normal dead time
        self.deadtime[~overflowmask] = 2.E-6  # s


def gbm_light_curve_plot(time_bins, cnts, bkg, width, selection):
    fig = plt.figure(777)
    ax = fig.add_subplot(111)

    max_cnts = max(cnts / width)
    top = max_cnts + max_cnts * .2
    min_cnts = min(cnts[cnts > 0] / width)
    bottom = min_cnts - min_cnts * .05
    mean_time = map(np.mean, time_bins)

    all_masks = []

    step_plot(time_bins, cnts / width, ax, color='#8da0cb', label="Light Curve")

    for tmin, tmax in selection:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:
            step_plot(time_bins[mask], cnts[mask] / width, ax,
                      color='#fc8d62',
                      fill=True,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width, ax,
              color='#fc8d62',
              fill=True,
              fill_min=min_cnts, label="Selection")

    ax.plot(mean_time, bkg, '#66c2a5', lw=2., label="Background")

    # ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)
    ax.legend()
