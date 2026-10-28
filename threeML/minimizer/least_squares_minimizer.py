import logging

import numpy as np
import scipy.optimize

from threeML.minimizer.minimization import FitFailed, LocalMinimizer

log = logging.getLogger(__name__)


_SUPPORTED_ALGORITHMS = ["trf", "dogbox"]

# Residual assigned to every bin when the trial values are not allowed
_PENALTY_RESIDUAL = 1e6


class LeastSquaresMinimizer(LocalMinimizer):
    """Levenberg-Marquardt-like minimizer based on
    scipy.optimize.least_squares.

    Instead of the -log(likelihood) it minimizes the sum of the squares of
    the deviance residuals provided by the plugins (see
    PluginPrototype.get_deviance_residuals), which is equivalent. The
    curvature is built from first derivatives of the residuals (J^T J), so
    it does not need second derivatives of the likelihood. Together with
    derivative steps proportional to the parameter values (like
    "xset delta" in XSPEC) this makes the fit robust for models with a
    limited numerical precision, such as many XSPEC models.

    The covariance matrix is the inverse of J^T J at the best fit, i.e. the
    Gauss-Newton approximation of the Hessian (as reported by XSPEC). Use
    JointLikelihood.get_errors for profile likelihood errors.

    Setup keys:

    * diff_step: relative step for the numerical derivatives (default 0.01)
    * ftol: tolerance on the relative change of the statistic (default 1e-8)
    * max_nfev: maximum number of iterations (default: scipy default)
    * algorithm: "trf" (default) or "dogbox"

    All the plugins must provide deviance residuals.
    """

    valid_setup_keys = ("diff_step", "ftol", "max_nfev", "algorithm")

    def __init__(self, function, parameters, verbosity=1, setup_dict=None):
        self._scale = None
        self._jacobian = None

        super(LeastSquaresMinimizer, self).__init__(
            function, parameters, verbosity, setup_dict
        )

    def _setup(self, user_setup_dict):
        if user_setup_dict is None:
            self._setup_dict = {
                "diff_step": 0.01,
                "ftol": 1e-8,
                "max_nfev": None,
                "algorithm": "trf",
            }

        else:
            if "algorithm" in user_setup_dict:
                self.set_algorithm(user_setup_dict["algorithm"])

            for key in user_setup_dict:
                self._setup_dict[key] = user_setup_dict[key]

    def set_algorithm(self, algorithm: str) -> None:
        """Set the algorithm for scipy.optimize.least_squares. Valid entries
        are "trf" and "dogbox"."""

        if algorithm not in _SUPPORTED_ALGORITHMS:
            log.error("Supported algorithms are %s" % (",".join(_SUPPORTED_ALGORITHMS)))

            raise AssertionError()

        self._setup_dict["algorithm"] = algorithm

    def _get_scales_and_bounds(self):
        x0 = []
        scale = []
        lower = []
        upper = []

        for (
            cur_value,
            cur_delta,
            cur_min,
            cur_max,
        ) in self._internal_parameters.values():
            lo = -np.inf if cur_min is None else cur_min
            hi = np.inf if cur_max is None else cur_max

            # Derivative steps are proportional to the value of the parameter, but
            # never smaller than the delta of the parameter (this matters for
            # parameters close to zero, e.g., in log scale). Use a fraction of the
            # range if both are zero

            delta = 0.0 if cur_delta is None else abs(cur_delta)

            this_scale = max(abs(cur_value), delta)

            if this_scale == 0:
                this_scale = 0.01 * (hi - lo) if np.isfinite(hi - lo) else 1.0

            # The algorithms need a starting point strictly inside the bounds

            if cur_value <= lo or cur_value >= hi:
                margin = 1e-6 * this_scale

                if np.isfinite(hi - lo):
                    margin = min(margin, 1e-6 * (hi - lo))

                cur_value = min(max(cur_value, lo + margin), hi - margin)

            x0.append(cur_value)
            scale.append(this_scale)
            lower.append(lo)
            upper.append(hi)

        return np.array(x0), np.array(scale), np.array(lower), np.array(upper)

    def _minimize(self):
        residual_function = getattr(self.function, "residuals", None)

        if residual_function is None:
            log.error(
                "The LEAST_SQUARES minimizer can only be used with JointLikelihood"
            )

            raise FitFailed()

        x0, scale, lower, upper = self._get_scales_and_bounds()

        n_residuals = [None]

        def wrapper(z):
            try:
                r = residual_function(*(z * scale))

            except NotImplementedError:
                log.error(
                    "The LEAST_SQUARES minimizer requires all the plugins to provide "
                    "deviance residuals (see PluginPrototype.get_deviance_residuals)"
                )

                raise

            if r is None:
                if n_residuals[0] is None:
                    raise FitFailed(
                        "The starting point of the fit is not in the allowed parameter "
                        "space"
                    )

                return np.full(n_residuals[0], _PENALTY_RESIDUAL)

            n_residuals[0] = r.shape[0]

            return r

        res = scipy.optimize.least_squares(
            wrapper,
            x0 / scale,
            jac="3-point",
            bounds=(lower / scale, upper / scale),
            method=self._setup_dict["algorithm"],
            diff_step=self._setup_dict["diff_step"],
            ftol=self._setup_dict["ftol"],
            xtol=None,
            gtol=None,
            max_nfev=self._setup_dict["max_nfev"],
        )

        if res.status < 0:
            raise FitFailed(
                "Could not converge. Message from solver: %s (status: %i)"
                % (res.message, res.status)
            )

        if res.status == 0:
            log.warning(
                "LEAST_SQUARES reached the maximum number of iterations before "
                "convergence: %s" % res.message
            )

        names = list(self.parameters.keys())

        for name, active in zip(names, res.active_mask):
            if active != 0:
                log.warning(
                    "Parameter %s is at its %s bound"
                    % (name, "lower" if active < 0 else "upper")
                )

        log.debug(
            "LEAST_SQUARES: %s (nfev=%s, njev=%s)" % (res.message, res.nfev, res.njev)
        )

        best_fit_values = res.x * scale

        self._scale = scale
        self._jacobian = res.jac

        # Evaluate the -log(likelihood) at the best fit (this also sets the
        # parameters to the best fit values)

        function_minimum = float(self.function(*best_fit_values))

        return best_fit_values, function_minimum

    def _compute_covariance_matrix(self, best_fit_values):
        """The covariance matrix is the inverse of the Gauss-Newton
        approximation of the Hessian of -log(likelihood), J^T J, where J
        is the Jacobian of the deviance residuals at the best fit."""

        n_dim = len(best_fit_values)

        if self._jacobian is None:
            return np.zeros((n_dim, n_dim)) * np.nan

        jtj = self._jacobian.T @ self._jacobian

        try:
            covariance_scaled = np.linalg.inv(jtj)

        except np.linalg.LinAlgError:
            log.warning(
                "Cannot invert the J^T J matrix, looks like the matrix is singular"
            )

            return np.zeros((n_dim, n_dim)) * np.nan

        return covariance_scaled * np.outer(self._scale, self._scale)
