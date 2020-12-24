import types
from collections.abc import Iterable
from typing import Optional, Tuple, Union

import astromodels
import numpy as np

from threeML.plugin_prototype import PluginPrototype

__instrument_name = "n.a."


class EventObservation(object):
    def __init__(
        self,
        events: np.ndarray,
        exposure: float,
        start: Union[float, np.ndarray],
        stop: Union[float, np.ndarray],
    ):

        self._events = np.array(events)
        self._exposure = exposure

        if isinstance(start, Iterable) or isinstance(stop, Iterable):

            assert isinstance(start, Iterable)
            assert isinstance(stop, Iterable)

            assert len(start) == len(stop)

            for i, v in enumerate(start):

                assert v < stop[i]

            self._start: np.ndarray = start

            self._stop: np.ndarray = stop

            self._is_multi_interval = True

        else:

            self._start: float = start

            self._stop: float = stop

            self._is_multi_interval = False

    @property
    def events(self) -> np.ndarray:
        return self._events

    @property
    def start(self) -> Union[float, np.ndarray]:
        return self._start

    @property
    def stop(self) -> Union[float, np.ndarray]:
        return self._stop

    @property
    def is_multi_interval(self) -> bool:
        return self._events


class UnbinnedPoissonLike(PluginPrototype):
    def __init__(
        self,
        name: str,
        observation: EventObservation,
        source_name: Optional[str] = None,
    ) -> None:

        assert isinstance(observation, EventObservation)

        self._observation: EventObservation = observation

        self._source_name = source_name

        super(UnbinnedPoissonLike, self).__init__(name=name, nuisance_parameters={})

    def set_model(self, model: astromodels.Model) -> None:
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._like_model = model

        # We assume there are no extended sources, since we cannot handle them here

        assert self._like_model.get_number_of_extended_sources() == 0, (
            "SpectrumLike plugins do not support " "extended sources"
        )

        # check if we set a source name that the source is in the model

        if self._source_name is not None:
            assert self._source_name in self._like_model.sources, (
                "Source %s is not contained in "
                "the likelihood model" % self._source_name
            )

        differential, integral = self._get_diff_flux_and_integral(self._like_model)

        self._integral_model = integral

        self._model = differential

    def _get_diff_and_integral(
        self, likelihood_model: astromodels.Model
    ) -> Tuple[types.FunctionType, types.FunctionType]:

        if self._source_name is None:

            n_point_sources = likelihood_model.get_number_of_point_sources()

            # Make a function which will stack all point sources (OGIP do not support spatial dimension)

            def differential(energies):
                fluxes = likelihood_model.get_point_source_fluxes(
                    0, energies, tag=self._tag
                )

                # If we have only one point source, this will never be executed
                for i in range(1, n_point_sources):
                    fluxes += likelihood_model.get_point_source_fluxes(
                        i, energies, tag=self._tag
                    )

                return fluxes

        else:

            # This SpectrumLike dataset refers to a specific source

            # Note that we checked that self._source_name is in the model when the model was set

            try:

                def differential_flux(energies):

                    return likelihood_model.sources[self._source_name](
                        energies, tag=self._tag
                    )

            except KeyError:

                raise KeyError(
                    "This plugin has been assigned to source %s, "
                    "which does not exist in the current model" % self._source_name
                )

        # New way with simpson rule.
        # Make sure to not calculate the model twice for the same energies
        def integral(e1, e2):
            # Simpson's rule
            # single energy values given
            return (
                (e2 - e1)
                / 6.0
                * (
                    differential(e1)
                    + 4 * differential((e2 + e1) / 2.0)
                    + differential(e2)
                )
            )

        return differential, integral

    def get_log_like(self) -> float:
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        n_expected_counts = 0

        if self._observation._is_multi_interval:

            for start, stop in zip(self._observation.start, self._observation.stop):

                n_expected_counts += self._integral_model(start, stop)

        else:

            n_expected_counts += self._integral_model(
                self._observation.start, self._observation.stop
            )

        M = self._model(self._observation.events) * self._observation.exposure

        return -n_expected_counts * np.log(M).sum()

    def inner_fit(self) -> float:
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()
