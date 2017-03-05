__author__ = 'giacomov'

import collections

import ROOT
import numpy as np

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tgraph_to_arrays, th2_to_arrays, tree_to_ndarray
from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIP.response import InstrumentResponse
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, BinnedSpectrum
from threeML.exceptions.custom_exceptions import custom_warnings

__instrument_name = "VERITAS"


# This is the data format v 1.0 agreed with Udara:
# * each run is in a different folder within the ROOT file, called run_XXXXXX
# * each run contains the following trees:

_trees_in_run = ['data_on',  # Event list in the source region
                 'data_off',  # Event list of the background region
                 'tRunSummary',  # Summary info on the run (exposure, and so on, see below)
                 'gMeanEffectiveArea',  # Effective area
                 'fAccZe_0',  # relative acceptance with respect to the on-axis area
                 'hMigration'  # Redistribution matrix (energy dispersion)
                 ]

# In the data_* trees we have:

_columns_in_data_tree = ['Time',  # it is the MJD of individual events
                         'Erec'  # reconstructed energy. *****Note that there are some events with negative
                                      # energy. Those are the events that we were unable to reconstruct energy.
                                      # In VERITAS when we calculate significance some times we use these events,
                                      # if we are not quoting the energy. Because failing the energy reconstruction
                                      # does not mean that the event is a bad event. However, if we quote an energy
                                      # band we do not use them.
                         'Xoff',  # this is the X offset from the detector centre in degrees.
                         'Yoff',  # this is the Y offset from the detector centre in degrees.
                         'theta2'  # X^2 + Y^2 (i.e., off-axis radius squared)
                         ]


class VERITASRun(object):

    def __init__(self, root_file, run_name):

        self._run_name = run_name

        # Read the data from the ROOT file

        with open_ROOT_file(root_file) as f:

            # Read first the TTrees as pandas DataFrame

            self._data_on = tree_to_ndarray(f.Get(run_name+'/data_on'))  # type: np.ndarray
            self._data_off = tree_to_ndarray(f.Get(run_name+'/data_off'))  # type: np.ndarray
            self._tRunSummary = np.squeeze(tree_to_ndarray(f.Get(run_name+'/tRunSummary')))  # type: np.ndarray

            # Now read the histogram
            self._log_mc_energies, \
            self._log_recon_energies, \
            self._hMigration = th2_to_arrays(f.Get(run_name + "/hMigration"))

            assert np.allclose(self._log_mc_energies, self._log_recon_energies)

            # Renormalize the migration matrix for the spectrum which was used to generate it
            # (a power law with index -1.5)
            w = (-2 * (10**self._log_mc_energies[1:])**(-0.5) + 2 * (10**self._log_mc_energies[:-1])**(-0.5))

            # Renormalize the migration matrix to 1 (it's supposed to be a probability)
            for i in range(self._hMigration.shape[0]):

                this_column = self._hMigration[i, :] / w  # type: np.ndarray

                renorm = np.sum(this_column)

                if renorm > 0:

                    self._hMigration[i, :] = this_column / renorm

            # Read the TGraph
            tgraph = f.Get(run_name + "/gMeanEffectiveArea")
            self._log_eff_area_energies, self._eff_area = tgraph_to_arrays(tgraph)

        # Exposure is tOn*(1-tDeadTimeFracOn)
        self._exposure = float(self._tRunSummary['DeadTimeFracOn']) * float(self._tRunSummary['tOn'])

        # Members for generating OGIP equivalents

        self._mission = "VERITAS"
        self._instrument = "VERITAS"

        # Now bin the counts
        self._counts, _ = np.histogram(self._data_on['Erec'], self._log_recon_energies)

        # Now bin the background counts
        self._bkg_counts, _ = np.histogram(self._data_off['Erec'], self._log_recon_energies)

        # Build the response
        self._response = self._build_response()

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

        repr += "%s src counts, %s bkg counts\n" % (np.sum(self._counts), np.sum(self._bkg_counts))

        repr += "Exposure: %.2f s, on area / off area: %.2f\n" % (self._exposure, float(self._tRunSummary['OffNorm']))

        print(repr)

    def _build_response(self):

        # Interpolate the effective area on the same bins of the migration matrix
        # NOTE: these are mid energies in log space
        mid_energies = (10**self._log_mc_energies[1:] + 10**self._log_mc_energies[:-1]) / 2.0  #type: np.ndarray

        log_mid_energies = np.log10(mid_energies)

        interpolated_effective_area = np.interp(log_mid_energies,
                                                self._log_eff_area_energies, self._eff_area,
                                                left=0, right=0)

        # Transform to cm2 from m2
        interpolated_effective_area *= 1e4

        # Get response matrix, which is effective area times energy dispersion
        matrix = self._hMigration * interpolated_effective_area

        # TODO: use energy dispersion
        # Avoid energy dispersion for the moment
        #matrix = np.identity(self._hMigration.shape[0], float) * interpolated_effective_area

        # Put a lower limit different than zero to avoid problems downstream when convolving a model with the response

        # Energies in VERITAS files are in TeV, we need them in keV

        response = InstrumentResponse(matrix, (10**self._log_recon_energies) * 1e9, (10**self._log_mc_energies) * 1e9)

        return response

    def get_spectrum(self):

        spectrum = BinnedSpectrumWithDispersion(counts=self._counts,
                                                exposure=self._exposure,
                                                response=self._response,
                                                is_poisson=True,
                                                mission=self._mission,
                                                instrument=self._instrument)

        return spectrum

    def get_background_spectrum(self):

        # Renormalization for the background (on_area / off_area), so this is usually < 1
        bkg_renorm = float(self._tRunSummary['OffNorm'])

        # by renormalizing the exposure of the background we account for the fact that the area is larger
        # (it is equivalent to introducing a renormalization)
        renormed_exposure = self._exposure / bkg_renorm

        background_spectrum = BinnedSpectrum(counts=self._bkg_counts,
                                             exposure=renormed_exposure,
                                             ebounds=self._response.ebounds,
                                             is_poisson=True,
                                             mission=self._mission,
                                             instrument=self._instrument)

        return background_spectrum


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

        run_names = filter(lambda x: x.find("run") == 0, keys)

        self._runs_like = collections.OrderedDict()

        for run_name in run_names:

            # Build the VERITASRun class
            this_run = VERITASRun(veritas_root_data, run_name)

            this_run.display()

            if this_run.total_counts == 0 or this_run.total_background_counts == 0:

                custom_warnings.warn("%s has 0 source or bkg counts, cannot use it." % run_name)
                continue

            else:

                # Get background spectrum and observation spectrum (with response)
                this_observation = this_run.get_spectrum()
                this_background = this_run.get_background_spectrum()

                self._runs_like[run_name] = DispersionSpectrumLike(run_name,
                                                                   this_observation,
                                                                   this_background)

        super(VERITASLike, self).__init__(name, {})

    def rebin_on_background(self, *args, **kwargs):

        for run in self._runs_like.values():

            run.rebin_on_background(*args, **kwargs)

    def rebin_on_source(self, *args, **kwargs):

        for run in self._runs_like.values():

            run.rebin_on_source(*args, **kwargs)

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        # Set the model for all runs
        for run in self._runs_like.values():

            run.set_model(likelihood_model_instance)

    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        # Collect the likelihood from each run
        total = 0

        for run in self._runs_like.values():

            total += run.get_log_like()

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



