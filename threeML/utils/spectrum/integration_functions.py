import numpy as np

from packaging.version import Version

np_version = Version(np.__version__)
if np_version < Version("2.0.0"):
    trapezoid = np.trapz
else:
    trapezoid = np.trapezoid


def _trapz(x, y):
    return trapezoid(x, y)


def _simps(e1, e2, diff_fluxes_edges, diff_fluxes_mid):
    return (
        (e2 - e1)
        / 6.0
        * (diff_fluxes_edges[:-1] + 4 * diff_fluxes_mid + diff_fluxes_edges[1:])
    )


def _rsum(model_mid_points, de):

    return np.multiply(model_mid_points, de)


def simpson_integral_edges(e_edges, diff_flux_method):

    e_m = (e_edges[1:] + e_edges[:-1]) / 2.0

    diff_fluxes_edges = diff_flux_method(e_edges)
    diff_fluxes_mid = diff_flux_method(e_m)

    return _simps(
        e_edges[:-1],
        e_edges[1:],
        diff_fluxes_edges,
        diff_fluxes_mid,
    )


def simpson_integral_values(e1, e2, diff_flux_method):
    # single energy values given
    return (
        (e2 - e1)
        / 6.0
        * (
            diff_flux_method(e1)
            + 4 * diff_flux_method((e2 + e1) / 2.0)
            + diff_flux_method(e2)
        )
    )


def trapz_integral_edges(e_edges, diff_flux_method):
    ee1 = e_edges[:-1]
    ee2 = e_edges[1:]

    diff_fluxes_edges = diff_flux_method(e_edges)

    return _trapz(
        np.array([diff_fluxes_edges[:-1], diff_fluxes_edges[1:]]).T,
        np.array([ee1, ee2]).T,
    )


def trapz_integral_values(e1, e2, diff_flux_method):
    # single energy values given
    return _trapz(
        np.array([diff_flux_method(e1), diff_flux_method(e2)]),
        np.array([e1, e2]),
    )


def riemann_integral_edges(e_edges, diff_flux_method):
    ee1 = e_edges[:-1]
    ee2 = e_edges[1:]

    e_m = (ee1 + ee2) / 2.0

    # energy width
    de = ee2 - ee1

    return _rsum(diff_flux_method(e_m), de)


def riemann_integral_values(e1, e2, diff_flux_method):
    return diff_flux_method(0.5 * (e1 + e2)) * (e2 - e1)
