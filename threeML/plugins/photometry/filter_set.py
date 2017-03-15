import collections
import numpy as np
import matplotlib.pyplot as plt

import pysynphot
import pysynphot.units as synphot_units

from threeML.io.plotting.cmap_cycle import cmap_intervals


class FilterSet(object):
    def __init__(self, filter_names, wave_lengths, transmission_curves, magnitude_systems, wavesunits='nm'):
        """

        :param filter_names: array of filter names
        :param wave_lengths: the wave lengths for each transmission curve
        :param transmission_curves: the transmission or throughput curves
        :param magnitude_systems: the magnitude system for each transmission curve
        :param wavesunits: the units of the wave length. see pysynphot for appropriate units http://pysynphot.readthedocs.io/en/latest/units.html?
        """



        self._transmission_curves = np.atleast_2d(transmission_curves)

        self._wave_lengths = np.atleast_2d(wave_lengths)

        self._magnitude_systems = magnitude_systems

        self._filter_names = np.array(filter_names)

        self._waveunits = wavesunits

        assert len(self._filter_names) == len(self._magnitude_systems), 'magnitude systems and filter names are different lengths'

        assert len(self._wave_lengths) == len(self._filter_names), 'wave lengths and filter names are different lengths'

        assert self._transmission_curves.shape == self._wave_lengths.shape, 'the wavenlengths and transmission curves are diffrent shapes'

        self._n_bands = len(filter_names)

        # build the filter curves with pysynphot

        self._build_filters()

        self._model_set = False


        # even when units are specified on the wave table
        # pysynphot operation return in anstroms. So we need a
        # converter

        self._angstrom = synphot_units.Angstrom()

    def _build_filters(self):

        # collect the band passes into an ordered dict

        self._bandpass = collections.OrderedDict()

        for name, wavelength, curve in zip(self._filter_names, self._wave_lengths, self._transmission_curves):
            bandpass = pysynphot.ArrayBandpass(wave=wavelength,
                                               throughput=curve,
                                               name=name,
                                               waveunits=self._waveunits
                                               )

            self._bandpass[name] = bandpass

    def set_model(self, differential_flux):
        """
        Wrap astromodels model into a pysynphot model and assign it to an observation
        """

        wrapper = AstromodelWrapper(differential_flux)

        self._observations = collections.OrderedDict()

        # now set the model for stimulus with the filter
        for k, v in self._bandpass.iteritems():
            self._observations[k] = pysynphot.Observation(wrapper, v)

        self._model_set = True

    def effective_stimulus(self):
        """
        return the effective stimulus of the model and filter for the given
        magnitude system
        :return:
        """

        assert self._model_set, 'no likelihood model has been set'

        return np.array(
            [obs.effstim(mag_sys) for obs, mag_sys in zip(self._observations.itervalues(), self._magnitude_systems)])

    def plot_filters(self):
        """
        plot the filter/ transmission curves
        :return: fig
        """

        fig, ax = plt.subplots()

        cc = cmap_intervals(self._n_bands, 'spectral')
        i = 0
        for key, band in self._bandpass.iteritems():
            ax.fill_between(band.wave,
                            0.,
                            band.throughput,
                            color=cc[i],
                            alpha=.8,
                            label=key)

            i += 1
        ax.legend()
        ax.set_ylabel('Throughput')
        ax.set_xlabel(self._waveunits)
        ax.set_ylim(bottom=0., top=1.)

        return fig

    @property
    def effective_widths(self):

        widths = np.array([self._angstrom.Convert(band.photbw(),self._waveunits) for band in self._bandpass.itervalues()])

        return widths

    @property
    def n_bands(self):
        """

        :return: the number of bands
        """

        return self._filter_names.shape[0]

    @property
    def filter_names(self):
        """

        :return: the filter names
        """

        return self._filter_names

    @property
    def average_wavelength(self):
        """

        :return: the average wave length of the filters
        """

        return [self._angstrom.Convert(band.avgwave(),self._waveunits) for band in self._bandpass.itervalues()]

    @property
    def waveunits(self):
        """

        :return: the pysynphot wave units
        """

        return self._waveunits


hc = 12.3984129 # keV A
h = 4.1356674E-18 # keV s

class AstromodelWrapper(pysynphot.spectrum.AnalyticSpectrum):
    def __init__(self, differential_flux):
        """
        wrap and astromodel call into the pysynphot
        model.
        """

        # we must shift A -> keV in the function input

        waveunits = pysynphot.units.Units('angstrom')  # go to keV

        # photolam units are pht /s /cm2/ A
        # thus, we must shift A -> keV for astromodels

        fluxunits = pysynphot.units.Units('photnu')  # make our own flux unit

        self._differential_flux = differential_flux

        super(AstromodelWrapper, self).__init__(waveunits, fluxunits)


    def __call__(self, wavelength):

        # the function argument from astromodels expects keV
        # but pysynphot will give us angstrom,
        # so we convert E = hc/lambda

        # the output of the function is expected to be :
        #     pht/ s / cm2 / Hz
        # so we convert keV to Hz via E = h nu




        return self._differential_flux(hc / wavelength) * h
