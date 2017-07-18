import copy
import pandas as pd
import numpy as np

from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, BinnedSpectrum
from threeML.plugins.OGIP.response import InstrumentResponse

from astromodels import PointSource, Model


__instrument_name = "General binned spectral data with energy dispersion"

class DispersionSpectrumLike(SpectrumLike):
    def __init__(self, name, observation, background=None, verbose=True):


        assert isinstance(observation, BinnedSpectrumWithDispersion), "observed spectrum is not an instance of BinnedSpectrumWithDispersion"

        assert observation.response is not None, "the observed spectrum does not have a response"

        # assign the response to the plugins

        self._rsp = observation.response #type: InstrumentResponse



        super(DispersionSpectrumLike, self).__init__(name=name,
                                                     observation=observation,
                                                     background=background,
                                                     verbose=verbose)



    def set_model(self, likelihoodModel):
        """
        Set the model to be used in the joint minimization.
        """

        # Store likelihood model

        self._like_model = likelihoodModel

        # We assume there are no extended sources, since we cannot handle them here

        assert self._like_model.get_number_of_extended_sources() == 0, "OGIP-like plugins do not support " \
                                                                       "extended sources"

        # Get the differential flux function, and the integral function, with no dispersion,
        # we simply integrate the model over the bins

        differential_flux, integral = self._get_diff_flux_and_integral()

        self._rsp.set_function(integral)

    def _evaluate_model(self):
        """
        evaluates the full model over all channels
        :return:
        """

        return self._rsp.convolve()

    def get_simulated_dataset(self, new_name=None,**kwargs):
        """
        Returns another DispersionSpectrumLike instance where data have been obtained by randomizing the current expectation from the
        model, as well as from the background (depending on the respective noise models)

        :return: a DispersionSpectrumLike simulated instance
         """

        # pass the response thru to the constructor
        return super(DispersionSpectrumLike, self).get_simulated_dataset(new_name=new_name,
                                                                         **kwargs)

    def get_pha_files(self):
        info = {}

        # we want to pass copies so that
        # the user doesn't grab the instance
        # and try to modify things. protection
        info['pha'] = copy.copy(self._observed_spectrum)
        info['bak'] = copy.copy(self._background_spectrum)
        info['rsp'] = copy.copy(self._rsp)

        return info

    def display_rsp(self):
        """
        Display the currently loaded full response matrix, i.e., RMF and ARF convolved
        :return:
        """

        self._rsp.plot_matrix()

    @property
    def response(self):

        return self._rsp

    def _output(self):
        # type: () -> pd.Series

        super_out = super(DispersionSpectrumLike, self)._output() #type: pd.Series

        the_df = pd.Series({'response':self._rsp.rsp_filename})


        return super_out.append(the_df)

    def write_pha(self, filename, overwrite=False, force_rsp_write=False):
        """
        Writes the observation, background and (optional) rsp to PHAII fits files

        :param filename: base file name to write out
        :param overwrite: if you would like to force overwriting of the files
        :param force_rsp_write: force the writing of an rsp even if not required

        """

        # we need to pass up the variables to an OGIPLike
        # so that we have the proper variable name

        # a local import here because OGIPLike is dependent on this

        from threeML.plugins.OGIPLike import OGIPLike

        ogiplike = OGIPLike.from_general_dispersion_spectrum(self)
        ogiplike.write_pha(file_name=filename, overwrite=overwrite, force_rsp_write=force_rsp_write)


    @classmethod
    def from_function(cls, name, source_function, response, source_errors=None, source_sys_errors=None,
                      background_function=None, background_errors=None, background_sys_errors=None):
        """

        Construct a simulated spectrum from a given source function and (optional) background function. If source and/or background errors are not supplied, the likelihood is assumed to be Poisson.

        :param name: simulkated data set name
        :param source_function: astromodels function
        :param response: 3ML Instrument response
        :param source_errors: (optional) gaussian source errors
        :param source_sys_errors: (optional) systematic source errors
        :param background_function: (optional) astromodels background function
        :param background_errors: (optional) gaussian background errors
        :param background_sys_errors: (optional) background systematic errors
        :return: simulated SpectrumLike plugin
        """



        # this is just for construction

        n_energies = response.ebounds.shape[0]-1





        fake_data = np.ones(n_energies)

        if source_errors is None:

            is_poisson = True

        else:

            assert len(source_errors) == n_energies, 'source error array is not the same dimension as the energy array'

            is_poisson = False

        if source_sys_errors is not None:
            assert len(source_sys_errors) == n_energies, 'background  systematic error array is not the same dimension as the energy array'

        observation = BinnedSpectrumWithDispersion(fake_data,
                                     exposure=1.,
                                     response=response,
                                     count_errors=source_errors,
                                     sys_errors=source_sys_errors,
                                     quality=None,
                                     scale_factor=1.,
                                     is_poisson=is_poisson,
                                     mission='fake_mission',
                                     instrument='fake_instrument',
                                     tstart=0.,
                                     tstop=1.)

        if background_function is not None:

            fake_background = np.ones(n_energies)

            if background_errors is None:

                is_poisson = True

            else:

                assert len(background_errors) == n_energies, 'background error array is not the same dimension as the energy array'

                is_poisson = False

            if background_sys_errors is not None:
                assert len(background_sys_errors) ==  n_energies, 'background  systematic error array is not the same dimension as the energy array'

            tmp_background = BinnedSpectrum(fake_background,
                                            exposure=1.,
                                            ebounds=response.ebounds,
                                            count_errors=background_errors,
                                            sys_errors=background_sys_errors,
                                            quality=None,
                                            scale_factor=1.,
                                            is_poisson=is_poisson,
                                            mission='fake_mission',
                                            instrument='fake_instrument',
                                            tstart=0.,
                                            tstop=1.)

            # now we have to generate the background counts
            # we treat the background as a simple observation with no
            # other background

            background_gen = SpectrumLike('generator', tmp_background, None, verbose=False)

            pts_background = PointSource("fake_background", 0.0, 0.0, background_function)

            background_model = Model(pts_background)

            background_gen.set_model(background_model)

            sim_background = background_gen.get_simulated_dataset('fake')

            background = sim_background._observed_spectrum


        else:

            background = None

        speclike_gen = cls('generator', observation, background, verbose=False)

        pts = PointSource("fake", 0.0, 0.0, source_function)

        model = Model(pts)

        speclike_gen.set_model(model)

        return speclike_gen.get_simulated_dataset(name)