__author__ = 'giacomov'

import numpy as np

import ROOT
import root_numpy

from threeML.io.cern_root_utils.io_utils import get_list_of_keys, open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tgraph_to_arrays, th2_to_arrays, tree_to_ndarray
from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.OGIP.pha import POISSON_PHAII
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIP.response import RMF
import astropy.io.fits as pyfits

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

        # Read the data from the ROOT file

        with open_ROOT_file(root_file) as f:

            # Read first the TTrees as pandas DataFrame

            self._data_on = tree_to_ndarray(f.Get(run_name+'/data_on'))  # type: np.ndarray
            self._data_off = tree_to_ndarray(f.Get(run_name+'/data_off'))  # type: np.ndarray
            self._tRunSummary = np.squeeze(tree_to_ndarray(f.Get(run_name+'/tRunSummary')))  # type: np.ndarray

            # Now read the histogram
            self._mc_energies, self._recon_energies, self._hMigration = th2_to_arrays(f.Get(run_name + "/hMigration"))

            # Read the TGraph
            tgraph = f.Get(run_name + "/gMeanEffectiveArea")
            self._gMeanEffectiveArea_energies, self._gMeanEffectiveArea_area = tgraph_to_arrays(tgraph)

        # Static members for genereting OGIP equivalents

        self._mission = "VERITAS"
        self._instrument = "VERITAS"

    def get_response(self, filename):

        rmf = RMF()

    def get_pha(self):

        # Get the exposure
        exposure = float(self._tRunSummary['DeadTimeFracOn'])

        # Bin the counts in energy
        counts, _ = np.histogram(self._data_on['Erec'], self._recon_energies)

        # Now get the rates
        rates = counts / exposure

        # Assume Poisson statistic
        rate_err = np.sqrt(counts) / exposure

        # Number of "channels"
        n_channels = self._recon_energies.shape[0]

        # G.V: here is how you could do this with the new frame work
        # so that you are not tied to PHA format

        #
        # spectrum = BinnedSpectrumWithDispersion(counts=counts,
        #                                         exposure=exposure,
        #                                         response=self._rsp_file,  # should be a fully formed InstrumentResponse
        #                                         is_poisson=True,
        #                                         mission=self._mission,
        #                                         instrument=self._instrument)
        #
        # return spectrum

        pha = POISSON_PHAII(instrument_name=self._instrument,
                    telescope_name=self._mission,
                    channel=,
                    rate=rates,
                    quality=np.zeros_like(rates, dtype=int),
                    grouping=np.ones(n_channels),
                    exposure=exposure,
                    backscale=None,
                    respfile=self._rsp_file,
                    ancrfile=None)

        return pha



class VERITASData(object):

    def __init__(self, veritas_root_data):

        # Open file

        f = ROOT.TFile(veritas_root_data)

        try:

            # Loop over the runs
            keys = get_list_of_keys(f)

            run_numbers = filter(lambda x:x.find("run") == 0, keys)

            for run in run_numbers:

                pass


        finally:

            f.close()


# Try dispersionLike?

# class VERITASLike(BinnedSpectrumWithDispersion):
#
#     def __init__(self, name, udara_style_root_file):
#
#
#         # here build your BinnedSpectraWithDispersion
#
#
#
#         super(VERITASLike, self).__init__(name,
#                                           observation, # will have response as a member
#                                           background, # might need to override if you do not have a bakcground
#         )
#

# This function get a list of

class VERITASLike(OGIPLike):

    def __init__(self, name, udara_style_root_file):

        super(VERITASLike, self).__init__(name, {})