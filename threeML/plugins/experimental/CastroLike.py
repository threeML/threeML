from __future__ import division
from past.utils import old_div
from builtins import object
from threeML.plugin_prototype import PluginPrototype
from threeML.exceptions.custom_exceptions import custom_warnings

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize

import matplotlib.pyplot as plt


class IntervalContainer(object):
    def __init__(
        self, start, stop, parameter_values, likelihood_values, n_integration_points
    ):

        # Make sure there is no NaN or infinity
        assert np.all(
            np.isfinite(likelihood_values)
        ), "Infinity or NaN in likelihood values"

        likelihood_values = np.asarray(likelihood_values)
        parameter_values = np.asarray(parameter_values)

        self._start = start
        self._stop = stop

        # Make sure the number of integration points is uneven, and that there are at minimum 11 points
        # n_integration_points = max(int(n_integration_points), 11)

        if n_integration_points % 2 == 0:

            # n_points is even, it shouldn't be otherwise things like Simpson rule will have problems
            n_integration_points += 1

            custom_warnings.warn(
                "The number of integration points should not be even. Adding +1"
            )

        self._n_integration_points = int(n_integration_points)

        # Build interpolation of the likelihood curve
        self._minus_likelihood_interp = scipy.interpolate.InterpolatedUnivariateSpline(
            np.log10(parameter_values), -likelihood_values, k=1, ext=0
        )

        # Find maximum of loglike
        idx = likelihood_values.argmax()

        self._min_par_value = parameter_values.min()
        self._max_par_value = parameter_values.max()

        res = scipy.optimize.minimize_scalar(
            self._minus_likelihood_interp,
            bounds=(np.log10(self._min_par_value), np.log10(self._max_par_value)),
            method="bounded",
            options={"maxiter": 10000, "disp": True, "xatol": 1e-3},
        )

        # res = scipy.optimize.minimize(self._minus_likelihood_interp, x0=[np.log10(parameter_values[idx])],
        #                               jac=lambda x:self._minus_likelihood_interp.derivative(1)(x),
        #                                      # bounds=(self._min_par_value, self._max_par_value),
        #                                      # method='bounded',
        #                                      tol=1e-3,
        #                                      options={'maxiter': 10000, 'disp': True})

        assert res.success, "Could not find minimum"

        self._minimum = (10 ** res.x, float(res.fun))

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def n_integration_points(self):
        return self._n_integration_points

    def __call__(self, parameter_value):

        return -self._minus_likelihood_interp(np.log10(parameter_value))

    def get_measurement(
        self,
        delta_log_like=0.5,
        ul_log_like=2.71 / 2.0,
        low_bound_extreme=0.0,
        hi_bound_extreme=np.inf,
    ):

        # Find when the likelihood changes by delta_log_like unit
        bounding_f = (
            lambda x: self._minus_likelihood_interp(np.log10(x))
            - self._minimum[1]
            - delta_log_like
        )

        if bounding_f(self._min_par_value) <= 0:

            # This is an upper limit measurement, i.e., there is no lower bound on the confidence. Use
            # low_bound_extreme
            low_bound_cl = low_bound_extreme

        else:

            # Look for negative bound using BRENTQ
            low_bound_cl, res = scipy.optimize.brentq(
                bounding_f, self._min_par_value, self._minimum[0], full_output=True
            )

            assert res.converged, "Could not find lower bound"

        if bounding_f(self._max_par_value) <= 0.0:

            # This is a lower limit measurement, i.e., there is no upper bound on the confidence. Use
            # hi_bound_extreme
            hi_bound_cl = hi_bound_extreme

        else:

            # If there was no lower limit, then compute upper bound for 95% confidence
            if low_bound_cl == low_bound_extreme:

                bounding_f = (
                    lambda x: self._minus_likelihood_interp(np.log10(x))
                    - self._minimum[1]
                    - ul_log_like
                )

            # Look for positive bound using BRENTQ
            hi_bound_cl, res = scipy.optimize.brentq(
                bounding_f, self._minimum[0], self._max_par_value, full_output=True
            )

            assert res.converged, "Could not find upper bound"

        return low_bound_cl, self._minimum[0], hi_bound_cl


class CastroLike(PluginPrototype):
    def __init__(self, name, interval_containers):

        self._interval_containers = sorted(interval_containers, key=lambda x: x.start)

        # By default all containers are active
        self._active_containers = self._interval_containers

        self._likelihood_model = None

        self._all_xx, self._all_xx_split, self._splits = self._setup_x_values()

        super(CastroLike, self).__init__(name, {})

    def _setup_x_values(self):

        # Create a list of all x values for each container
        xxs = []
        splits = []

        # This will keep the total number of x values, so we can check for overlapping intervals
        total_n = 0

        # Loop over the active containers and fill the list
        for container in self._active_containers:

            xxs.append(
                np.logspace(
                    np.log10(container.start),
                    np.log10(container.stop),
                    container.n_integration_points,
                )
            )

            total_n += container.n_integration_points

            splits.append(total_n)

        all_xx = np.concatenate(xxs)

        assert (
            all_xx.shape[0] == total_n
        ), "One or more containers are overlapping. This is not supported."

        return all_xx, np.split(all_xx, splits), splits

    def set_active_measurements(self, tmin, tmax):

        self._active_containers = []

        for interval_container in self._interval_containers:

            if interval_container.start >= tmin and interval_container.stop <= tmax:

                self._active_containers.append(interval_container)

        # Reset the global xx
        self._all_xx, self._all_xx_split, self._splits = self._setup_x_values()

        return len(self._active_containers)

    @property
    def start(self):
        return min([x.start for x in self._active_containers])

    @property
    def stop(self):
        return max([x.stop for x in self._active_containers])

    @property
    def active_containers(self):
        return self._active_containers

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._likelihood_model = likelihood_model_instance

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        log_l = 0.0

        # Evaluate once for all
        all_yy = np.split(
            self._likelihood_model.get_total_flux(self._all_xx), self._splits
        )

        for i, interval_container in enumerate(self._active_containers):

            # Get integral of model between start and stop for this interval
            # xx = np.logspace(np.log10(interval_container.start),
            #                  np.log10(interval_container.stop),
            #                  interval_container.n_integration_points)
            #
            # yy = self._likelihood_model.get_total_flux(xx)
            #
            # assert np.allclose(yy, all_yy[i])
            xx = self._all_xx_split[i]
            yy = all_yy[i]

            length = interval_container.stop - interval_container.start

            expected_flux = old_div(scipy.integrate.simps(yy, xx), length)

            this_log_l = interval_container(expected_flux)

            log_l += this_log_l

        return log_l

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()

    @staticmethod
    def _plot(containers, sub, color):

        xs = []
        xerrs = []
        ys = []
        yerrs = [[], []]

        uls_xs = []
        uls_xerrs = []
        uls_ys = []
        uls_yerrs = []

        for interval_container in containers:

            t1, t2 = interval_container.start, interval_container.stop
            tc = (t2 + t1) / 2.0
            dt = (t2 - t1) / 2.0

            y_low, y, y_hi = interval_container.get_measurement()

            if y_low > 0.0:

                # A normal point

                xs.append(tc)
                xerrs.append(dt)

                ys.append(y)
                yerrs[0].append((y - y_low))
                yerrs[1].append((y_hi - y))

            else:

                # Upper limit
                uls_xs.append(tc)
                uls_xerrs.append(dt)
                uls_ys.append(y_hi)

                # Make an errorbar that is constant length in log space
                dy_ = np.log10(y_hi) - 0.2
                dy = y_hi - 10 ** dy_

                uls_yerrs.append(dy)

        sub.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt=",", ecolor=color, mfc=color)

        sub.errorbar(
            uls_xs,
            uls_ys,
            xerr=uls_xerrs,
            yerr=uls_yerrs,
            uplims=True,
            fmt=",",
            ecolor=color,
            mfc=color,
        )

        xxs = np.append(xs, uls_xs)
        xxerrs = np.append(xerrs, uls_xerrs)

        idx = np.argsort(xxs)
        xxs = xxs[idx]
        xxerrs = xxerrs[idx]

        return xxs, xxerrs

    def plot(self, plot_model=True, n_points=1000, fig=None, sub=None):

        if fig is None:

            fig, sub = plt.subplots()

        else:

            if sub is None:

                assert len(fig.axes) > 0

                sub = fig.axes[0]

        xs, xerrs = self._plot(self._active_containers, sub, "blue")

        # Find out which containers are not active
        inactive_containers = []

        for container in self._interval_containers:

            if container not in self._active_containers:

                inactive_containers.append(container)

        if len(inactive_containers) > 0:

            self._plot(inactive_containers, sub, "gray")

        sub.set_xscale("log")
        sub.set_yscale("log")

        if plot_model:

            xs = np.asarray(xs)
            xerrs = np.asarray(xerrs)

            min_idx = xs.argmin()
            max_idx = xs.argmax()

            xx = np.logspace(
                np.log10(xs[min_idx] - xerrs[min_idx]),
                np.log10(xs[max_idx] + xerrs[max_idx]),
                n_points,
            )

            yy = self._likelihood_model.get_total_flux(xx)

            _ = sub.plot(xx, yy, linestyle="--", color="red")

        return fig
