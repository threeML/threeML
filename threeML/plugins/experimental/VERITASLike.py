from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import collections

import ROOT
import numpy as np

import scipy.integrate
import astromodels

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import (
    tgraph_to_arrays,
    th2_to_arrays,
    tree_to_ndarray,
)
from threeML.plugin_prototype import PluginPrototype
from threeML.exceptions.custom_exceptions import custom_warnings

from threeML.utils.statistics.likelihood_functions import (
    poisson_observed_poisson_background,
)

__instrument_name = "VERITAS"


# Integrate the interpolation of the effective area for each bin in the residstribution matrix, then sum over the MC
# energies for the same bin, then renormalize the latter to be the same. The factor should be the same for all
# channels


# This is the data format v 1.0 agreed with Udara:
# * each run is in a different folder within the ROOT file, called run_XXXXXX
# * each run contains the following trees:

_trees_in_run = [
    "data_on",  # Event list in the source region
    "data_off",  # Event list of the background region
    "tRunSummary",  # Summary info on the run (exposure, and so on, see below)
    "gMeanEffectiveArea",  # Effective area
    "fAccZe_0",  # relative acceptance with respect to the on-axis area
    "hMigration",  # Redistribution matrix (energy dispersion)
]

# In the data_* trees we have:

_columns_in_data_tree = [
    "Time",  # it is the MJD of individual events
    "Erec"  # reconstructed energy. *****Note that there are some events with negative
    # energy. Those are the events that we were unable to reconstruct energy.
    # In VERITAS when we calculate significance some times we use these events,
    # if we are not quoting the energy. Because failing the energy reconstruction
    # does not mean that the event is a bad event. However, if we quote an energy
    # band we do not use them.
    "Xoff",  # this is the X offset from the detector centre in degrees.
    "Yoff",  # this is the Y offset from the detector centre in degrees.
    "theta2",  # X^2 + Y^2 (i.e., off-axis radius squared)
]


class VERITASRun(object):
    def __init__(self, root_file, run_name):

        self._run_name = run_name

        # Read the data from the ROOT file

        with open_ROOT_file(root_file) as f:

            # Read first the TTrees as pandas DataFrame

            self._data_on = tree_to_ndarray(
                f.Get(run_name + "/data_on")
            )  # type: np.ndarray
            self._data_off = tree_to_ndarray(
                f.Get(run_name + "/data_off")
            )  # type: np.ndarray
            self._tRunSummary = np.squeeze(
                tree_to_ndarray(f.Get(run_name + "/tRunSummary"))
            )  # type: np.ndarray

            # Now read the histogram
            (
                self._log_recon_energies,
                self._log_mc_energies,
                self._hMigration,
            ) = th2_to_arrays(f.Get(run_name + "/hMigration"))

            # Transform energies to keV (they are in TeV)
            self._log_recon_energies += 9
            self._log_mc_energies += 9

            # Compute bin centers and bin width of the Monte Carlo energy bins

            self._dE = (
                10 ** self._log_mc_energies[1:] - 10 ** self._log_mc_energies[:-1]
            )
            self._mc_energies_c = (
                10 ** self._log_mc_energies[1:] + 10 ** self._log_mc_energies[:-1]
            ) / 2.0
            self._recon_energies_c = (
                10 ** self._log_recon_energies[1:] + 10 ** self._log_recon_energies[:-1]
            ) / 2.0

            self._n_chan = self._log_recon_energies.shape[0] - 1

            # Remove all nans by substituting them with 0.0
            idx = np.isfinite(self._hMigration)
            self._hMigration[~idx] = 0.0

            # Read the TGraph
            tgraph = f.Get(run_name + "/gMeanEffectiveArea")
            self._log_eff_area_energies, self._eff_area = tgraph_to_arrays(tgraph)

            # Transform the effective area to cm2 (it is in m2 in the file)
            self._eff_area *= (
                1e8  # This value is for VEGAS, because VEGAS effective area is in cm2
            )

            # Transform energies to keV
            self._log_eff_area_energies += 9

        # Now use the effective area provided in the file to renormalize the migration matrix appropriately
        self._renorm_hMigration()

        # Exposure is tOn*(1-tDeadtimeFrac)
        self._exposure = float(1 - self._tRunSummary["DeadTimeFracOn"]) * float(
            self._tRunSummary["tOn"]
        )

        # Members for generating OGIP equivalents

        self._mission = "VERITAS"
        self._instrument = "VERITAS"

        # Now bin the counts

        self._counts, _ = self._bin_counts_log(
            self._data_on["Erec"] * 1e9, self._log_recon_energies
        )

        # Now bin the background counts

        self._bkg_counts, _ = self._bin_counts_log(
            self._data_off["Erec"] * 1e9, self._log_recon_energies
        )

        print(
            "Read a %s x %s matrix, spectrum has %s bins, eff. area has %s elements"
            % (
                self._hMigration.shape[0],
                self._hMigration.shape[1],
                self._counts.shape[0],
                self._eff_area.shape[0],
            )
        )

        # Read in the background renormalization (ratio between source and background region)

        self._bkg_renorm = float(self._tRunSummary["OffNorm"])

        self._start_energy = np.log10(175e6)  # 175 GeV in keV
        self._end_energy = np.log10(18e9)  # 18 TeV in keV
        self._first_chan = (
            np.abs(self._log_recon_energies - self._start_energy)
        ).argmin()
        self._last_chan = (np.abs(self._log_recon_energies - self._end_energy)).argmin()

    def _renorm_hMigration(self):

        # Get energies where the effective area is given

        energies_eff = 10 ** self._log_eff_area_energies

        # Get the unnormalized effective area x photon flux contained in the migration matrix

        v = np.sum(self._hMigration, axis=0)

        # Get the expected photon flux using the simulated spectrum

        mc_e1 = 10 ** self._log_mc_energies[:-1]
        mc_e2 = 10 ** self._log_mc_energies[1:]

        rc_e1 = 10 ** self._log_recon_energies[:-1]
        rc_e2 = 10 ** self._log_recon_energies[1:]

        expectation = self._simulated_spectrum(self._recon_energies_c) * (rc_e2 - rc_e1)

        # Get the unnormalized effective area

        # Compute the renormalization based on the energy range from 200 GeV to 1 TeV

        emin = 0.2 * 1e9
        emax = 1 * 1e9

        # idx = (self._mc_energies_c > emin) & (self._mc_energies_c < emax)
        # avg1 = np.average(new_v[idx])

        # idx = (energies_eff > emin) & (energies_eff < emax)
        # avg2 = np.average(self._eff_area[idx])

        # renorm = avg1 / avg2

        # Added by for bin by bin normalization
        v_new = np.sum(self._hMigration, axis=1)
        new_v = old_div(v_new, expectation)
        avg1_new = new_v
        avg2_new = np.interp(self._recon_energies_c, energies_eff, self._eff_area)
        renorm_new = old_div(avg1_new, avg2_new)
        hMigration_new = old_div(self._hMigration, renorm_new[:, None])
        hMigration_new[~np.isfinite(hMigration_new)] = 0

        self._hMigration = hMigration_new

    @staticmethod
    def _bin_counts_log(counts, log_bins):

        energies_on_log = np.log10(np.array(counts))

        # Substitute nans (due to negative energies in unreconstructed events)

        energies_on_log[~np.isfinite(energies_on_log)] = -99

        return np.histogram(energies_on_log, log_bins)

    @property
    def migration_matrix(self):

        return self._hMigration

    @property
    def total_counts(self):

        return np.sum(self._counts)

    @property
    def total_background_counts(self):

        return np.sum(self._bkg_counts)

    def display(self):

        repr = "%s:\n" % self._run_name

        repr += "%s src counts, %s bkg counts\n" % (
            np.sum(self._counts),
            np.sum(self._bkg_counts),
        )

        repr += "Exposure: %.2f s, on area / off area: %.2f\n" % (
            self._exposure,
            float(self._tRunSummary["OffNorm"]),
        )

        failed_on_idx = self._data_on["Erec"] <= 0
        failed_off_idx = self._data_off["Erec"] <= 0

        repr += "Events with failed reconstruction: %i src, %i bkg" % (
            np.sum(failed_on_idx),
            np.sum(failed_off_idx),
        )

        print(repr)

    def _get_diff_flux_and_integral(self, like_model):

        n_point_sources = like_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (OGIP do not support spatial dimension)

        def differential_flux(energies):
            fluxes = like_model.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):
                fluxes += like_model.get_point_source_fluxes(i, energies)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):
            # Simpson's rule

            return (
                (e2 - e1)
                / 6.0
                * (
                    differential_flux(e1)
                    + 4 * differential_flux((e1 + e2) / 2.0)
                    + differential_flux(e2)
                )
            )

        return differential_flux, integral

    @staticmethod
    def _simulated_spectrum(x):

        return (x) ** (-2.45)

    @staticmethod
    def _simulated_spectrum_f(e1, e2):

        integral_f = lambda x: old_div(-3.0, (x ** 0.5))

        return integral_f(e2) - integral_f(e1)

    @staticmethod
    def _integrate(function, e1, e2):

        integrals = []

        for ee1, ee2 in zip(e1, e2):

            grid = np.linspace(ee1, ee2, 30)

            integrals.append(scipy.integrate.simps(function(grid), grid))

        # integrals = map(lambda x:scipy.integrate.quad(function, x[0], x[1], epsrel=1e-2)[0], zip(e1, e2))

        return np.array(integrals)

    def get_log_like(self, like_model, fast=True):

        # Reweight the response matrix
        diff_flux, integral = self._get_diff_flux_and_integral(like_model)

        e1 = 10 ** self._log_mc_energies[:-1]
        e2 = 10 ** self._log_mc_energies[1:]

        dE = e2 - e1

        if not fast:

            this_spectrum = old_div(
                self._integrate(diff_flux, e1, e2), dE
            )  # 1 / keV cm2 s

            sim_spectrum = old_div(
                self._simulated_spectrum_f(e1, e2), dE
            )  # 1 / keV cm2 s

        else:
            this_spectrum = diff_flux(self._mc_energies_c)

            sim_spectrum = self._simulated_spectrum(self._mc_energies_c)

        weight = old_div(this_spectrum, sim_spectrum)  # type: np.ndarray

        # print("Sum of weight: %s" % np.sum(weight))

        n_pred = np.zeros(self._n_chan)

        for i in range(n_pred.shape[0]):

            n_pred[i] = np.sum(self._hMigration[i, :] * weight) * self._exposure

        log_like, _ = poisson_observed_poisson_background(
            self._counts, self._bkg_counts, self._bkg_renorm, n_pred
        )
        log_like_tot = np.sum(
            log_like[self._first_chan : self._last_chan + 1]
        )  # type: float

        # print("%s: obs: %s, npred: %s, bkg: %s (%s), npred + bkg: %s -> %.2f" % (self._run_name,
        #                                                              np.sum(self._counts),
        #                                                              np.sum(n_pred),
        #                                                              np.sum(self._bkg_counts),
        #                                                              np.sum(self._bkg_counts) * self._bkg_renorm,
        #                                             np.sum(n_pred)+ np.sum(self._bkg_counts) * self._bkg_renorm,
        #                                                                        log_like_tot))

        return log_like_tot, locals()


class VERITASLike(PluginPrototype):
    def __init__(self, name, veritas_root_data):

        # Open file

        f = ROOT.TFile(veritas_root_data)

        try:

            # Loop over the runs
            keys = get_list_of_keys(f)

        finally:

            f.Close()

        # Get the names of all runs included

        run_names = [x for x in keys if x.find("run") == 0]

        self._runs_like = collections.OrderedDict()

        for run_name in run_names:

            # Build the VERITASRun class
            this_run = VERITASRun(veritas_root_data, run_name)

            this_run.display()

            if this_run.total_counts == 0 or this_run.total_background_counts == 0:

                custom_warnings.warn(
                    "%s has 0 source or bkg counts, cannot use it." % run_name
                )
                continue

            else:

                # Get background spectrum and observation spectrum (with response)
                # this_observation = this_run.get_spectrum()
                # this_background = this_run.get_background_spectrum()
                #
                # self._runs_like[run_name] = DispersionSpectrumLike(run_name,
                #                                                    this_observation,
                #                                                    this_background)
                #
                # self._runs_like[run_name].set_active_measurements("c50-c130")
                self._runs_like[run_name] = this_run

        super(VERITASLike, self).__init__(name, {})

    def rebin_on_background(self, *args, **kwargs):

        for run in list(self._runs_like.values()):

            run.rebin_on_background(*args, **kwargs)

    def rebin_on_source(self, *args, **kwargs):

        for run in list(self._runs_like.values()):

            run.rebin_on_source(*args, **kwargs)

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        # Set the model for all runs
        self._likelihood_model = likelihood_model_instance  # type: astromodels.Model

        # for run in self._runs_like.values():
        #
        #     run.set_model(likelihood_model_instance)

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        # Collect the likelihood from each run
        total = 0

        for run in list(self._runs_like.values()):

            total += run.get_log_like(self._likelihood_model)[0]

        return total

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()
