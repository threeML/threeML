import collections
import numpy as np
import matplotlib.pyplot as plt

import pysynphot


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


        assert transmission_curves.shape == wave_lengths.shape

        assert len(filter_names) == len(magnitude_systems)

        assert len(wave_lengths) == len(filter_names)

        self._filter_names = np.array(filter_names)

        self._wavelengths = wave_lengths

        self._waveunits = wavesunits

        self._transmission_curves = transmission_curves

        self._magnitude_systems = magnitude_systems

        self._n_bands = len(filter_names)

        # build the filter curves with pysynphot

        self._build_filters()

        self._model_set = False

    def _build_filters(self):

        # collect the band passes into an ordered dict

        self._bandpass = collections.OrderedDict()

        for name, wavelength, curve in zip(self._filter_names, self._wavelengths, self._transmission_curves):

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


        assert self._model_set

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

        return [band.avgwave() for band in self._bandpass.itervalues()]

    @property
    def waveunits(self):
        """

        :return: the pysynphot wave units
        """

        return self._waveunits


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

        fluxunits = pysynphot.units.Units('photlam')  # make our own flux unit

        self._differential_flux = differential_flux

        super(AstromodelWrapper, self).__init__(waveunits, fluxunits)

        self._Angstrom_to_keV = 1239842.5

    def __call__(self, x):
        return self._differential_flux(x * self._Angstrom_to_keV) / self._Angstrom_to_keV