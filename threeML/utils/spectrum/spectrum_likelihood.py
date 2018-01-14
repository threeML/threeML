from threeML.utils.statistics.likelihood_functions import half_chi2
from threeML.utils.statistics.likelihood_functions import poisson_log_likelihood_ideal_bkg
from threeML.utils.statistics.likelihood_functions import poisson_observed_gaussian_background
from threeML.utils.statistics.likelihood_functions import poisson_observed_poisson_background


import numpy as np

class SpectrumLikelihood(object):

    def __init__(self, spectrum_plugin):

        self._spectrum_plugin = spectrum_plugin


    def get_log_likelihood(self):

        RuntimeError('must be implemented in subclass')


class GaussianObservedLikelihood(SpectrumLikelihood):


    def get_log_likelihood(self):

        chi2_ = half_chi2(self._spectrum_plugin.current_observed_counts,
                           self._spectrum_plugin.current_observed_count_errors,
                           self._spectrum_plugin.get_model() )

        assert np.all(np.isfinite(chi2_))

        return np.sum(chi2_) * (-1), None


class PoissonObservedIdealBackgroundLikelihood(SpectrumLikelihood):

    def get_log_likelihood(self):

        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model()

        loglike, _ = poisson_log_likelihood_ideal_bkg(self._spectrum_plugin.current_observed_counts,
                                                      self._spectrum_plugin.current_scaled_background_counts,
                                                      model_counts)

        return np.sum(loglike), None

class PoissonObservedModeledBackgroundLikelihood(SpectrumLikelihood):

    def get_log_likelihood(self):

        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model()

        # we scale the background model to the observation

        background_model_counts = self._spectrum_plugin.get_background_model() * self._spectrum_plugin._scale_factor

        loglike, _ = poisson_log_likelihood_ideal_bkg(self._spectrum_plugin.current_observed_counts,
                                                      background_model_counts,
                                                      model_counts)

        bkg_log_like, _ = self._spectrum_plugin.background_plugin.get_log_like()

        total_log_like = np.sum(loglike) + bkg_log_like

        return total_log_like, None

class PoissonObservedNoBackgroundLikelihood(SpectrumLikelihood):

    def get_log_likelihood(self):

        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model()

        background_model_counts = np.zeros_like(model_counts)

        loglike, _ = poisson_log_likelihood_ideal_bkg(self._spectrum_plugin.current_observed_counts,
                                                      background_model_counts,
                                                      model_counts)

        return np.sum(loglike), None

class PoissonObservedPoissonBackgroundLikelihood(SpectrumLikelihood):

    def get_log_likelihood(self):
        # Scale factor between source and background spectrum

        model_counts = self._spectrum_plugin.get_model()

        loglike, bkg_model = poisson_observed_poisson_background(self._spectrum_plugin.current_observed_counts,
                                                                 self._spectrum_plugin.current_background_counts,
                                                                 self._spectrum_plugin.scale_factor,
                                                                 model_counts)

        return np.sum(loglike), bkg_model

class PoissonObservedGaussianBackgroundLikelihood(SpectrumLikelihood):

    def get_log_likelihood(self):

        expected_model_counts = self._spectrum_plugin.get_model()

        loglike, bkg_model = poisson_observed_gaussian_background(self._spectrum_plugin.current_observed_counts,
                                                                  self._spectrum_plugin.current_background_counts,
                                                                  self._spectrum_plugin.current_back_count_errors,
                                                                  expected_model_counts)

        return np.sum(loglike), bkg_model


likelihood_lookup = {'poisson':{'poisson' : PoissonObservedPoissonBackgroundLikelihood,
                                'gaussian': PoissonObservedGaussianBackgroundLikelihood,
                                'ideal' : PoissonObservedIdealBackgroundLikelihood,
                                 None : PoissonObservedNoBackgroundLikelihood,
                                'modeled': PoissonObservedModeledBackgroundLikelihood


                                },

                     'gaussian':{None : GaussianObservedLikelihood}

                     }