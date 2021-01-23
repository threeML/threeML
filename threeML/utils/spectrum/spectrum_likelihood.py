from builtins import object
import copy
from typing import Optional

import numpy as np
import numba as nb


from threeML.utils.numba_utils import nb_sum
from threeML.utils.statistics.likelihood_functions import half_chi2
from threeML.utils.statistics.likelihood_functions import (
    poisson_log_likelihood_ideal_bkg,
)
from threeML.utils.statistics.likelihood_functions import (
    poisson_observed_gaussian_background,
)
from threeML.utils.statistics.likelihood_functions import (
    poisson_observed_poisson_background,
)

from threeML.io.logging import setup_logger

log = setup_logger(__name__)

# These classes provide likelihood evaluation to SpectrumLike and children

_known_noise_models = {}


class BinnedStatistic(object):
    def __init__(self, spectrum_plugin):
        """
        
        A class to hold the likelihood call and randomization of spectrum counts
        
        :param spectrum_plugin: the spectrum plugin to call
        """

        self._spectrum_plugin = spectrum_plugin


    def get_current_value(self):
        RuntimeError("must be implemented in subclass")

    def get_randomized_source_counts(self, source_model_counts):
        return None

    def get_randomized_source_errors(self):
        return None

    def get_randomized_background_counts(self):
        return None

    def get_randomized_background_errors(self):
        return None


class GaussianObservedStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        
        model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        chi2_ = half_chi2(
            self._spectrum_plugin.current_observed_counts,
            self._spectrum_plugin.current_observed_count_errors,
            model_counts,
        )

        assert np.all(np.isfinite(chi2_))

        return nb_sum(chi2_) * (-1), None

    def get_randomized_source_counts(self, source_model_counts):
        idx = self._spectrum_plugin.observed_count_errors > 0

        randomized_source_counts = np.zeros_like(source_model_counts)

        randomized_source_counts[idx] = np.random.normal(
            loc=source_model_counts[idx],
            scale=self._spectrum_plugin.observed_count_errors[idx],
        )

        # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

        idx = randomized_source_counts < 0  # type: np.ndarray

        negative_source_n = nb_sum(idx)

        if negative_source_n > 0:
            log.warning(
                "Generated source has negative counts "
                "in %i channels. Fixing them to zero" % (negative_source_n)
            )

            randomized_source_counts[idx] = 0

        return randomized_source_counts

    def get_randomized_source_errors(self):
        return self._spectrum_plugin.observed_count_errors


class PoissonObservedIdealBackgroundStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        loglike, _ = poisson_log_likelihood_ideal_bkg(
            self._spectrum_plugin.current_observed_counts,
            self._spectrum_plugin.current_scaled_background_counts,
            model_counts,
        )

        return nb_sum(loglike), None

    def get_randomized_source_counts(self, source_model_counts):
        # Randomize expectations for the source
        # we want the unscalled background counts

        # TODO: check with giacomo if this is correct!

        randomized_source_counts = np.random.poisson(
            source_model_counts + self._spectrum_plugin._background_counts
        )

        return randomized_source_counts

    def get_randomized_background_counts(self):
        # No randomization for the background in this case

        randomized_background_counts = self._spectrum_plugin._background_counts

        return randomized_background_counts


class PoissonObservedModeledBackgroundStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        # we scale the background model to the observation

        background_model_counts = (
            self._spectrum_plugin.get_background_model()
            * self._spectrum_plugin.scale_factor
        )

        loglike, _ = poisson_log_likelihood_ideal_bkg(
            self._spectrum_plugin.current_observed_counts,
            background_model_counts,
            model_counts,
        )

        bkg_log_like = self._spectrum_plugin.background_plugin.get_log_like()

        total_log_like = nb_sum(loglike) + bkg_log_like

        return total_log_like, None

    def get_randomized_source_counts(self, source_model_counts):
        # first generate random source counts from the plugin

        self._synthetic_background_plugin = (
            self._spectrum_plugin.background_plugin.get_simulated_dataset()
        )

        randomized_source_counts = np.random.poisson(
            source_model_counts + self._synthetic_background_plugin.observed_counts
        )

        return randomized_source_counts

    def get_randomized_background_errors(self):
        randomized_background_count_err = None

        if not self._synthetic_background_plugin.observed_spectrum.is_poisson:
            randomized_background_count_err = (
                self._synthetic_background_plugin.observed_count_errors
            )

        return randomized_background_count_err

    @property
    def synthetic_background_plugin(self):
        return self._synthetic_background_plugin


class PoissonObservedNoBackgroundStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        # In this likelihood the background becomes part of the model, which means that
        # the uncertainty in the background is completely neglected

        model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        background_model_counts = np.zeros_like(model_counts)

        loglike, _ = poisson_log_likelihood_ideal_bkg(
            self._spectrum_plugin.current_observed_counts,
            background_model_counts,
            model_counts,
        )

        return nb_sum(loglike), None

    def get_randomized_source_counts(self, source_model_counts):
        # Randomize expectations for the source
        # we want the unscalled background counts

        randomized_source_counts = np.random.poisson(source_model_counts)

        return randomized_source_counts


class PoissonObservedPoissonBackgroundStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        # Scale factor between source and background spectrum
        model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        loglike, bkg_model = poisson_observed_poisson_background(
            self._spectrum_plugin.current_observed_counts,
            self._spectrum_plugin.current_background_counts,
            self._spectrum_plugin.scale_factor,
            model_counts,
        )

        return nb_sum(loglike), bkg_model

    def get_randomized_source_counts(self, source_model_counts):
        # Since we use a profile likelihood, the background model is conditional on the source model, so let's
        # get it from the likelihood function

        _, background_model_counts = self.get_current_value()

        # Now randomize the expectations

        # Randomize expectations for the source

        randomized_source_counts = np.random.poisson(
            source_model_counts + background_model_counts
        )

        return randomized_source_counts

    def get_randomized_background_counts(self):
        # Randomize expectations for the background

        _, background_model_counts = self.get_current_value()

        randomized_background_counts = np.random.poisson(background_model_counts)

        return randomized_background_counts


class PoissonObservedGaussianBackgroundStatistic(BinnedStatistic):
    def get_current_value(self, precalc_fluxes: Optional[np.array]=None):
        expected_model_counts = self._spectrum_plugin.get_model(precalc_fluxes=precalc_fluxes)

        loglike, bkg_model = poisson_observed_gaussian_background(
            self._spectrum_plugin.current_observed_counts,
            self._spectrum_plugin.current_background_counts,
            self._spectrum_plugin.current_background_count_errors,
            expected_model_counts,
        )

        return nb_sum(loglike), bkg_model

    def get_randomized_source_counts(self, source_model_counts):
        # Since we use a profile likelihood, the background model is conditional on the source model, so let's
        # get it from the likelihood function

        _, background_model_counts = self.get_current_value()

        # Now randomize the expectations

        # Randomize expectations for the source

        randomized_source_counts = np.random.poisson(
            source_model_counts + background_model_counts
        )

        return randomized_source_counts

    def get_randomized_background_counts(self):
        # Now randomize the expectations.

        _, background_model_counts = self.get_current_value()

        # We cannot generate variates with zero sigma. They variates from those channel will always be zero
        # This is a limitation of this whole idea. However, remember that by construction an error of zero
        # it is only allowed when the background counts are zero as well.
        idx = self._spectrum_plugin.background_count_errors > 0

        randomized_background_counts = np.zeros_like(background_model_counts)

        randomized_background_counts[idx] = np.random.normal(
            loc=background_model_counts[idx],
            scale=self._spectrum_plugin.background_count_errors[idx],
        )

        # Issue a warning if the generated background is less than zero, and fix it by placing it at zero

        idx = randomized_background_counts < 0  # type: np.ndarray

        negative_background_n = nb_sum(idx)

        if negative_background_n > 0:
            log.warning(
                "Generated background has negative counts "
                "in %i channels. Fixing them to zero" % (negative_background_n)
            )

            randomized_background_counts[idx] = 0

        return randomized_background_counts

    def get_randomized_background_errors(self):
        return copy.copy(self._spectrum_plugin.background_count_errors)


statistic_lookup = {
    "poisson": {
        "poisson": PoissonObservedPoissonBackgroundStatistic,
        "gaussian": PoissonObservedGaussianBackgroundStatistic,
        "ideal": PoissonObservedIdealBackgroundStatistic,
        None: PoissonObservedNoBackgroundStatistic,
        "modeled": PoissonObservedModeledBackgroundStatistic,
    },
    "gaussian": {None: GaussianObservedStatistic},
    None: {None: None},
}


