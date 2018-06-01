import numpy as np

from threeML.io.cern_root_utils.io_utils import open_ROOT_file
from threeML.io.cern_root_utils.tobject_to_numpy import tree_to_ndarray, th2_to_arrays
from threeML.utils.OGIP.response import InstrumentResponse



class POLARData(object):
    def __init__(self, polar_root_file, reference_time=0., rsp_file=None):
        """
        container class that converts raw POLAR root data into useful python
        variables


        :param polar_root_file: path to polar event file
        :param reference_time: reference time of the events (tunix?)
        :param rsp_file: path to rsp file
        """



        with open_ROOT_file(rsp_file) as f:
            matrix = th2_to_arrays(f.Get('rsp'))[-1]
            ebounds = th2_to_arrays(f.Get('EM_bounds'))[-1]
            mc_low = th2_to_arrays(f.Get('ER_low'))[-1]
            mc_high = th2_to_arrays(f.Get('ER_high'))[-1]



        mc_energies = np.append(mc_low, mc_high[-1])

        # open the event file
        with open_ROOT_file(polar_root_file) as f:
            tmp = tree_to_ndarray(f.Get('polar_out'))

            # extract the pedestal corrected ADC channels
            # which are non-integer and possibly
            # less than zero
            pha = tmp['Energy']

            # non-zero ADC channels are invalid
            idx = pha >= 0
            #pha = pha[idx]

            idx2 = (pha <= ebounds.max()) & (pha >= ebounds.min())

            pha = pha[idx2 & idx]

            # get the dead time fraction
            self._dead_time_fraction = tmp['dead_ratio'][idx & idx2]

            # get the arrival time, in tunix of the events
            self._time = tmp['tunix'][idx & idx2] - reference_time

            # digitize the ADC channels into bins
            # these bins are preliminary


        # build the POLAR response

        self._rsp = InstrumentResponse(matrix=matrix,
                                       ebounds=ebounds,
                                       monte_carlo_energies=mc_energies)

        # bin the ADC channels

        self._binned_pha = np.digitize(pha, ebounds)

    @property
    def pha(self):
        return self._binned_pha

    @property
    def time(self):
        return self._time

    @property
    def dead_time_fraction(self):
        return self._dead_time_fraction

    @property
    def rsp(self):
        return self._rsp

    @property
    def n_channels(self):

        return len(self._rsp.ebounds) - 1
