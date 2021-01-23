from __future__ import division

import copy
import warnings
from builtins import map, object, range, str
from operator import attrgetter, itemgetter
from pathlib import Path
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from matplotlib.colors import SymLogNorm
from past.utils import old_div
import numba as nb

from threeML.exceptions.custom_exceptions import custom_warnings
from threeML.io.file_utils import (file_existing_and_readable,
                                   fits_file_existing_and_readable,
                                   sanitize_filename)
from threeML.io.fits_file import FITSExtension, FITSFile
from threeML.utils.time_interval import TimeInterval, TimeIntervalSet

from threeML.io.logging import setup_logger

log = setup_logger(__name__)

class NoCoverageIntervals(RuntimeError):
    pass


class NonContiguousCoverageIntervals(RuntimeError):
    pass


class NoMatrixForInterval(RuntimeError):
    pass


class IntervalOfInterestNotCovered(RuntimeError):
    pass


class GapInCoverageIntervals(RuntimeError):
    pass


class InstrumentResponse(object):
    def __init__(self, matrix, ebounds, monte_carlo_energies, coverage_interval=None):
        """

        Generic response class that accepts a full matrix, detector energy boundaries (ebounds) and monte carlo energies,
        and an optional coverage interval which indicates which time interval the matrix applies to.

        If there are n_channels in the detector, and the monte carlo energies are n_mc_energies, then the matrix must
        be n_channels x n_mc_energies.

        Therefore, an OGIP style RSP from a file is not required if the matrix,
        ebounds, and mc channels exist.


        :param matrix: an n_channels x n_mc_energies response matrix representing both effective area and
        energy dispersion effects
        :param ebounds: the energy boundaries of the detector channels (size n_channels + 1)
        :param monte_carlo_energies: the energy boundaries of the monte carlo channels (size n_mc_energies + 1)
        :param coverage_interval: the time interval to which the matrix refers to (if available, None by default)
        :type coverage_interval: TimeInterval
        """

        # we simply store all the variables to the class

        self._matrix = np.array(matrix, float)
        self._matrix_transpose = self._matrix.T
        # Make sure there are no nans or inf
        assert np.all(np.isfinite(self._matrix)), "Infinity or nan in matrix"

        self._ebounds = np.array(ebounds, float)

        self._mc_energies = np.array(monte_carlo_energies)

        self._integral_function = None

        # Store the time interval
        if coverage_interval is not None:

            assert isinstance(
                coverage_interval, TimeInterval
            ), "The coverage interval must be a TimeInterval instance"

            self._coverage_interval = coverage_interval

        else:

            self._coverage_interval = None

        # Safety checks
        assert self._matrix.shape == (
            self._ebounds.shape[0] - 1,
            self._mc_energies.shape[0] - 1,
        ), "Matrix has the wrong shape. Got %s, expecting %s" % (
            self._matrix.shape,
            [self._ebounds.shape[0] - 1, self._mc_energies.shape[0] - 1],
        )

        if self._mc_energies.max() < self._ebounds.max():

            log.warning(
                "Maximum MC energy (%s) is smaller "
                "than maximum EBOUNDS energy (%s)"
                % (self._mc_energies.max(), self.ebounds.max()),
                #RuntimeWarning,
            )

        if self._mc_energies.min() > self._ebounds.min():

            log.warning(
                "Minimum MC energy (%s) is larger than "
                "minimum EBOUNDS energy (%s)"
                % (self._mc_energies.min(), self._ebounds.min()),
             #   RuntimeWarning,
            )

    # This will be overridden by subclasses
    @property
    def rsp_filename(self):
        """
        Returns the name of the RSP/RMF file from which the response has been loaded
        """

        return None

    # This will be overridden by subclasses
    @property
    def arf_filename(self):
        """
        Returns the name of the ARF file (or None if there is none)
        """

        return None

    @property
    def first_channel(self):

        # This is needed to write to PHA files. We use always 1 (and consistently we always use 1 in the MATRIX files
        # too, to avoid confusion (and because XSpec default is 1)

        return 1

    @property
    def coverage_interval(self):
        """
        Returns the time interval that this matrix is applicable to. None if it wasn't defined and the matrix is
        applicable everywhere

        :return time_interval: the time interval
        :type time_interval : TimeInterval
        """

        return self._coverage_interval

    @property
    def matrix(self):
        """
        Return the matrix representing the response

        :return matrix: response matrix
        :type matrix: np.ndarray
        """
        return self._matrix

    def replace_matrix(self, new_matrix):
        """
        Replace the read matrix with a new one of the same shape

        :return: none
        """

        assert new_matrix.shape == self._matrix.shape

        self._matrix = new_matrix
        self._matrix_transpose = self._matrix.T

    @property
    def ebounds(self):
        """

        Returns the ebounds of the RSP.

        :return:
        """
        return self._ebounds

    @property
    def monte_carlo_energies(self):
        """
        Returns the boundaries of the Monte Carlo bins (true energy bins)

        :return: array
        """

        return self._mc_energies

    def set_function(self, integral_function=None):
        """
        Set the function to be used for the convolution

        :param integral_function: a function f = f(e1,e2) which returns the integral of the model between e1 and e2
        :type integral_function: callable
        """

        self._integral_function = integral_function


    def convolve(self, precalc_fluxes: Optional[np.array]=None) -> np.ndarray:
        """
        Convolve the source flux with the response
        :param precalc_fluxes: The precalulated flux. If this is None, the
        flux gets calculated here.
        """
        if precalc_fluxes is None:
            fluxes = self._integral_function(
               # self._mc_energies[:-1], self._mc_energies[1:]
            )
        else:
            fluxes = precalc_fluxes

        # Sometimes some channels have 0 lenths, or maybe they start at 0, where
        # many functions (like a power law) are not defined. In the response these
        # channels have usually a 0, but unfortunately for a computer
        # inf * zero != zero. Thus, let's force this. We avoid checking this situation
        # in details because this would have a HUGE hit on performances

        idx = np.isfinite(fluxes)
        fluxes[~idx] = 0

        folded_counts = np.dot(fluxes, self._matrix_transpose)

        return folded_counts

    def energy_to_channel(self, energy):

        """Finds the channel containing the provided energy.
        NOTE: returns the channel index (starting at zero),
        not the channel number (likely starting from 1).

        If you ask for a energy lower than the minimum ebounds, 0 will be returned
        If you ask for a energy higher than the maximum ebounds, the last channel index will be returned
        """

        # Get the index of the first ebounds upper bound larger than energy
        # (but never go below zero or above the last channel)
        idx = min(
            max(0, np.searchsorted(self._ebounds, energy) - 1), len(self._ebounds) - 1
        )

        return idx

    def plot_matrix(self):

        fig, ax = plt.subplots()

        idx_mc = 0
        idx_eb = 0

        # Some times the lower edges may be zero, so we skip them

        if self._mc_energies[0] == 0:
            idx_mc = 1

        if self._ebounds[0] == 0:
            idx_eb = 1

        # ax.imshow(image[idx_eb:, idx_mc:], extent=(self._ebounds[idx_eb],
        #                                            self._ebounds[-1],
        #                                            self._mc_energies[idx_mc],
        #                                            self._mc_energies[-1]),
        #           aspect='equal',
        #           cmap=cm.BrBG_r,
        #           origin='lower',
        #           norm=SymLogNorm(1.0, 1.0, vmin=self._matrix.min(), vmax=self._matrix.max()))

        # Find minimum non-zero element
        vmin = self._matrix[self._matrix > 0].min()

        cmap = copy.deepcopy(cm.ocean)

        cmap.set_under("gray")

        mappable = ax.pcolormesh(
            self._mc_energies[idx_mc:],
            self._ebounds[idx_eb:],
            self._matrix,
            cmap=cmap,
            norm=SymLogNorm(1.0, 1.0, vmin=vmin, vmax=self._matrix.max()),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.colorbar(mappable, label="cm$^{2}$")

        # if show_energy is not None:
        #    ener_val = Quantity(show_energy).to(self.reco_energy.unit).value
        #    ax.hlines(ener_val, 0, 200200, linestyles='dashed')

        ax.set_xlabel("True energy (keV)")
        ax.set_ylabel("Reco energy (keV)")

        return fig

    def to_fits(self, filename, telescope_name, instrument_name, overwrite=False):
        """
        Write the current matrix into a OGIP FITS file

        :param filename : the name of the FITS file to be created
        :type filename : str
        :param telescope_name : a name for the telescope/experiment which this matrix applies to
        :param instrument_name : a name for the instrument which this matrix applies to
        :param overwrite: True or False, whether to overwrite or not the output file
        :return: None
        """

        filename: Path = sanitize_filename(filename, abspath=True)

        fits_file = RSP(
            self.monte_carlo_energies,
            self.ebounds,
            self.matrix,
            telescope_name,
            instrument_name,
        )

        fits_file.writeto(filename, overwrite=overwrite)

    @classmethod
    def create_dummy_response(cls, ebounds, monte_carlo_energies):
        """
        Creates a dummy identity response of the shape of the ebounds and mc energies

        :param ebounds: the energy boundaries of the detector channels (size n_channels + 1)
        :param monte_carlo_energies: the energy boundaries of the monte carlo channels (size n_mc_energies + 1)
        :return: InstrumentResponse
        """

        # create the dummy matrix

        dummy_matrix = np.eye(ebounds.shape[0] - 1, monte_carlo_energies.shape[0] - 1)

        return cls(dummy_matrix, ebounds, monte_carlo_energies)


class OGIPResponse(InstrumentResponse):
    def __init__(self, rsp_file, arf_file=None):
        """

        :param rsp_file:
        :param arf_file:
        """

        # Now make sure that the response file exist

        rsp_file: Path = sanitize_filename(rsp_file)

        assert fits_file_existing_and_readable(rsp_file), (
            f"OGIPResponse file {rsp_file} not existing or not " "readable"
        )

        # Check if we are dealing with a .rsp2 file (containing more than
        # one response). This is checked by looking for the syntax
        # [responseFile]{[responseNumber]}

        if "{" in str(rsp_file):

            tokens = str(rsp_file).split("{")
            rsp_file: Path = sanitize_filename(tokens[0])
            rsp_number = int(tokens[-1].split("}")[0].replace(" ", ""))

        else:

            rsp_number = 1

        self._rsp_file: Path = rsp_file

        # Read the response
        with pyfits.open(rsp_file) as f:

            try:

                # This is usually when the response file contains only the energy dispersion

                data = f["MATRIX", rsp_number].data
                header = f["MATRIX", rsp_number].header

                if arf_file is None:
                    log.warning(
                        "The response is in an extension called MATRIX, which usually means you also "
                        "need an ancillary file (ARF) which you didn't provide. You should refer to the "
                        "documentation  of the instrument and make sure you don't need an ARF."
                    )

            except Exception as e:
                log.warning(
                    "The default choice for MATRIX extension failed:"
                    + repr(e)
                    + "available: "
                    + " ".join([repr(e.header.get("EXTNAME")) for e in f])
                )

                # Other detectors might use the SPECRESP MATRIX name instead, usually when the response has been
                # already convoluted with the effective area

                # Note that here we are not catching any exception, because
                # we have to fail if we cannot read the matrix

                data = f["SPECRESP MATRIX", rsp_number].data
                header = f["SPECRESP MATRIX", rsp_number].header

            # These 3 operations must be executed when the file is still open

            matrix = self._read_matrix(data, header)

            ebounds = self._read_ebounds(f["EBOUNDS"])

            mc_channels = self._read_mc_channels(data)

        # Now, if there is information on the coverage interval, let's use it

        header_start = header.get("TSTART", None)
        header_stop = header.get("TSTOP", None)

        if header_start is not None and header_stop is not None:

            super(OGIPResponse, self).__init__(
                matrix=matrix,
                ebounds=ebounds,
                monte_carlo_energies=mc_channels,
                coverage_interval=TimeInterval(header_start, header_stop),
            )

        else:

            super(OGIPResponse, self).__init__(
                matrix=matrix, ebounds=ebounds, monte_carlo_energies=mc_channels
            )

        # Read the ARF if there is any
        # NOTE: this has to happen *after* calling the parent constructor

        if arf_file is not None and str(arf_file).lower() != "none":

            self._read_arf_file(arf_file)

        else:

            self._arf_file = None

    @staticmethod
    def _are_contiguous(arr1, arr2):

        return np.allclose(arr1[1:], arr2[:-1])

    def _read_ebounds(self, ebounds_extension):
        """
        reads the ebounds from an OGIP response

        :param ebounds_extension: an RSP ebounds extension
        :return:
        """

        e_min = ebounds_extension.data.field("E_MIN").astype(float)
        e_max = ebounds_extension.data.field("E_MAX").astype(float)

        assert self._are_contiguous(e_min, e_max), "EBOUNDS channel are not contiguous!"

        # The returned array must have the edges of the intervals. Doing so reduces the amount of memory used
        # by 1/2
        ebounds = np.append(e_min, [e_max[-1]])

        return ebounds

    def _read_mc_channels(self, data):
        """
        reads the mc_channels from an OGIP response

        :param data: data from a RSP MATRIX
        :return:
        """

        # Check for proper channels
        energ_lo = data.field("ENERG_LO").astype(float)
        energ_hi = data.field("ENERG_HI").astype(float)

        assert self._are_contiguous(
            energ_lo, energ_hi
        ), "Monte carlo channels are not contiguous"

        # The returned array must have the edges of the intervals. Doing so reduces the amount of memory used
        # by 1/2
        mc_channels = np.append(energ_lo, [energ_hi[-1]])

        return mc_channels

    @property
    def first_channel(self):
        """
        The first channel of the channel array. Corresponds to
        TLMIN keyword in FITS files

        :return: first channel
        """
        return int(self._first_channel)

    def _read_matrix(self, data, header, column_name="MATRIX"):

        n_channels = header.get("DETCHANS")

        assert (
            n_channels is not None
        ), "Matrix is improperly formatted. No DETCHANS keyword."

        # The header contains a keyword which tells us the first legal channel. It is TLMIN of the F_CHAN column
        # NOTE: TLMIN keywords start at 1, so TLMIN1 is the minimum legal value for the first column. So we need
        # to add a +1 because of course the numbering of lists (data.columns.names) starts at 0

        f_chan_column_pos = data.columns.names.index("F_CHAN") + 1

        try:
            tlmin_fchan = header["TLMIN%i" % f_chan_column_pos]

        except (KeyError):
            log.warning(
                "No TLMIN keyword found. This DRM does not follow OGIP standards. Assuming TLMIN=1"
            )
            tlmin_fchan = 1

        # Store the first channel as a property
        self._first_channel = tlmin_fchan

        rsp = np.zeros([data.shape[0], n_channels], float)

        n_grp = data.field("N_GRP")  # type: np.ndarray

        # The numbering of channels could start at 0, or at some other number (usually 1). Of course the indexing
        # of arrays starts at 0. So let's offset the F_CHAN column to account for that

        f_chan = data.field("F_CHAN") - tlmin_fchan  # type: np.ndarray
        n_chan = data.field("N_CHAN")  # type: np.ndarray

        # In certain matrices where compression has not been used, n_grp, f_chan and n_chan are not array columns,
        # but simple scalars. Expand then their dimensions so that we don't need to customize the code below.
        # However, if the columns are variable-length arrays, then they do have ndmin = 1 but have dtype 'object'.
        # In that case we don't want to add a dimension, as they are essentially a list of arrays.

        if n_grp.ndim == 1 and data.field("N_CHAN").dtype != np.object:
            n_grp = np.expand_dims(n_grp, 1)

        if f_chan.ndim == 1 and data.field("N_CHAN").dtype != np.object:
            f_chan = np.expand_dims(f_chan, 1)

        if n_chan.ndim == 1 and data.field("N_CHAN").dtype != np.object:
            n_chan = np.expand_dims(n_chan, 1)

        matrix = data.field(column_name)

        for i, row in enumerate(data):

            m_start = 0

            for j in range(np.squeeze(n_grp[i])):

                # This np.squeeze call is needed because some files (for example from Fermi/GBM) contains a vector
                # column for n_chan, even though all elements are of size 1
                this_n_chan = int(np.squeeze(n_chan[i][j]))
                this_f_chan = int(np.squeeze(f_chan[i][j]))

                rsp[i, this_f_chan : this_f_chan + this_n_chan] = matrix[i][
                    m_start : m_start + this_n_chan
                ]

                m_start += this_n_chan

        return rsp.T

    @property
    def rsp_filename(self) -> Path:
        """
        Returns the name of the RSP/RMF file from which the response has been loaded
        """

        return self._rsp_file

    @property
    def arf_filename(self):
        """
        Returns the name of the ARF file (or None if there is none)
        """

        return self._arf_file

    def _read_arf_file(self, arf_file: str):
        """
        read an arf file and apply it to the current_matrix

        :param arf_file:
        :param current_matrix:
        :param current_mc_channels:
        :return:
        """

        arf_file: Path = sanitize_filename(arf_file)

        self._arf_file = arf_file

        assert fits_file_existing_and_readable(arf_file), (
            f"Ancillary file {arf_file} not existing or not " "readable"
        )

        with pyfits.open(arf_file) as f:

            data = f["SPECRESP"].data

        arf = data.field("SPECRESP")

        # Check that arf and rmf have same dimensions

        if arf.shape[0] != self.matrix.shape[1]:
            raise IOError(
                "The ARF and the RMF file does not have the same number of channels"
            )

        # Check that the ENERG_LO and ENERG_HI for the RMF and the ARF
        # are the same

        energ_lo = data.field("ENERG_LO")
        energ_hi = data.field("ENERG_HI")

        assert self._are_contiguous(
            energ_lo, energ_hi
        ), "Monte carlo energies in ARF are not contiguous!"

        arf_mc_channels = np.append(energ_lo, [energ_hi[-1]])

        # Declare the mc channels different if they differ by more than 1%

        idx = self.monte_carlo_energies > 0

        diff = old_div(
            (self.monte_carlo_energies[idx] - arf_mc_channels[idx]),
            self.monte_carlo_energies[idx],
        )

        if diff.max() > 0.01:
            raise IOError(
                "The ARF and the RMF have one or more MC channels which differ by more than 1%"
            )

        # Multiply ARF and RMF

        matrix = self.matrix * arf

        # Override the matrix with the one multiplied by the arf
        self.replace_matrix(matrix)


class InstrumentResponseSet(object):
    """
    A set of responses

    """

    def __init__(self, matrix_list, exposure_getter, counts_getter, reference_time=0.0):
        """

        :param matrix_list:
        :type matrix_list : list[InstrumentResponse]
        :param exposure_getter : a function returning the exposure between t1 and t2
        :param counts_getter : a function returning the number of counts between t1 and t2
        :param reference_time : a reference time to be added to the specifications of the intervals used in the
        weight_by_* methods. Use this if you want to express the time intervals in time units from the reference_time,
        instead of "absolute" time. For GRBs, this is the trigger time. NOTE: if you use a reference time, the
        counts_getter and the exposure_getter must accept times relative to the reference time.
        """

        # Store list of matrices

        self._matrix_list = list(matrix_list)  # type: list[InstrumentResponse]

        # Create the corresponding list of coverage intervals

        self._coverage_intervals = TimeIntervalSet(
            [x.coverage_interval for x in self._matrix_list]
        )

        # Make sure that all matrices have coverage interval set

        if None in self._coverage_intervals:

            raise NoCoverageIntervals(
                "You need to specify the coverage interval for all matrices in the matrix_list"
            )

        # Remove from the list matrices that cover intervals of zero duration (yes, the GBM publishes those too,
        # one example is in data/ogip_test_gbm_b0.rsp2)
        to_be_removed = []
        for i, interval in enumerate(self._coverage_intervals):

            if interval.duration == 0:

                # Remove it
                with custom_warnings.catch_warnings():

                    custom_warnings.simplefilter("always", RuntimeWarning)

                    log.warning(
                        "Removing matrix %s (numbering starts at zero) because it has a coverage of "
                        "zero seconds" % i,
                        #RuntimeWarning,
                    )

                to_be_removed.append(i)

        # Actually remove them
        if len(to_be_removed) > 0:

            [self._matrix_list.pop(index) for index in to_be_removed]
            [self._coverage_intervals.pop(index) for index in to_be_removed]

        # Order the matrices by time

        idx = self._coverage_intervals.argsort()

        # It is possible that there is only one coverage interval (these are published by GBM e.g. GRB090819607)
        # so we need to be sure that the array is a least 1D

        self._coverage_intervals = TimeIntervalSet(
            np.atleast_1d(itemgetter(*idx)(self._coverage_intervals))
        )
        self._matrix_list = np.atleast_1d(itemgetter(*idx)(self._matrix_list))
        # Now make sure that the coverage intervals are contiguous (i.e., there are no gaps)
        if not self._coverage_intervals.is_contiguous():

            raise NonContiguousCoverageIntervals(
                "The provided responses have coverage intervals which are not contiguous!"
            )

        # Apply the reference time shift, if any
        self._coverage_intervals -= reference_time

        # Store callable

        self._exposure_getter = exposure_getter  # type: callable

        self._counts_getter = counts_getter  # type: callable

        # Store reference time

        self._reference_time = float(reference_time)

    @property
    def reference_time(self):

        return self._reference_time

    def __getitem__(self, item):

        return self._matrix_list[item]

    def __len__(self):

        return len(self._matrix_list)

    @classmethod
    def from_rsp2_file(
        cls,
        rsp2_file,
        exposure_getter,
        counts_getter,
        reference_time=0.0,
        half_shifted=True,
    ):

        # This assumes the Fermi/GBM rsp2 file format

        # make the rsp file proper
        rsp_file = sanitize_filename(rsp2_file)

        assert file_existing_and_readable(rsp_file), (
            "OGIPResponse file %s not existing or not readable" % rsp_file
        )

        # Will fill up the list of matrices
        list_of_matrices = []

        # Read the response
        with pyfits.open(rsp_file) as f:

            n_responses = f["PRIMARY"].header["DRM_NUM"]

            # we will read all the matrices and save them
            for rsp_number in range(1, n_responses + 1):

                this_response = OGIPResponse(str(rsp2_file) + "{%i}" % rsp_number)

                list_of_matrices.append(this_response)

        if half_shifted:

            # Now the GBM format has a strange feature: the matrix, instead of covering from TSTART to TSTOP, covers
            # from (TSTART + TSTOP) / 2.0 of the previous matrix to the (TSTART + TSTOP) / 2.0 of itself.
            # So let's adjust the coverage intervals accordingly

            if len(list_of_matrices) > 1:

                for i, this_matrix in enumerate(list_of_matrices):

                    if i == 0:

                        # The first matrix covers from its TSTART to its half time

                        this_matrix._coverage_interval = TimeInterval(
                            this_matrix.coverage_interval.start_time,
                            this_matrix.coverage_interval.half_time,
                        )

                    else:

                        # Any other matrix covers from the half time of the previous matrix to its half time
                        # However, the previous matrix has been already processed, so we use its stop time which
                        # has already begun the half time of what it was before processing

                        prev_matrix = list_of_matrices[i - 1]

                        this_matrix._coverage_interval = TimeInterval(
                            prev_matrix.coverage_interval.stop_time,
                            this_matrix.coverage_interval.half_time,
                        )

        return InstrumentResponseSet(
            list_of_matrices, exposure_getter, counts_getter, reference_time
        )

    # I didn't re-implement this at the moment
    # def _display_response_weighting(self, weights, tstarts, tstops):
    #
    #     fig, ax = plt.subplots()
    #
    #     # plot the time intervals
    #
    #     ax.hlines(min(weights) - .1, tstarts, tstops, color='red', label='selected intervals')
    #
    #     ax.hlines(np.median(weights), self._true_rsp_intervals[0], self._true_rsp_intervals[1], color='green',
    #               label='true rsp intervals')
    #
    #     ax.hlines(max(self._weight) + .1, self._matrix_start, self._matrix_stop, color='blue',
    #               label='rsp header intervals')
    #
    #     mean_true_rsp_time = np.mean(self._true_rsp_intervals.T, axis=1)
    #
    #     ax.plot(mean_true_rsp_time, self._weight, '+k', label='weight')

    def weight_by_exposure(self, *intervals):

        return self._get_weighted_matrix("exposure", *intervals)

    def weight_by_counts(self, *intervals):

        return self._get_weighted_matrix("counts", *intervals)

    def _get_weighted_matrix(self, switch, *intervals):

        assert len(intervals) > 0, "You have to provide at least one interval"

        intervals_set = TimeIntervalSet.from_strings(*intervals)

        # Compute a set of weights for each interval
        weights = np.zeros(len(self._matrix_list))

        for interval in intervals_set:

            weights += self._weight_response(interval, switch)

        # Normalize to 1
        weights /= np.sum(weights)

        # Weight matrices
        matrix = np.dot(
            np.array(list(map(attrgetter("matrix"), self._matrix_list))).T, weights.T
        ).T

        # Now generate the instance of the response

        # get EBOUNDS from the first matrix
        ebounds = self._matrix_list[0].ebounds

        # Get mc channels from the first matrix
        mc_channels = self._matrix_list[0].monte_carlo_energies

        matrix_instance = InstrumentResponse(matrix, ebounds, mc_channels)

        return matrix_instance

    def _weight_response(self, interval_of_interest, switch):

        """

        :param interval_start : start time of the interval
        :param interval_stop : stop time of the interval
        :param switch: either 'counts' or 'exposure'

        """

        #######################
        # NOTE: the weights computed here are *not* normalized to one so that they can be combined if there is
        # more than one interval
        #######################

        # Now mark all responses which overlap with the interval of interest
        # NOTE: this is a mask of the same length as _matrix_list and _coverage_intervals

        matrices_mask = [
            c_i.overlaps_with(interval_of_interest) for c_i in self._coverage_intervals
        ]

        # Check that we have at least one matrix

        if not np.any(matrices_mask):

            raise NoMatrixForInterval(
                "Could not find any matrix applicable to %s\n Have intervals:%s"
                % (
                    interval_of_interest,
                    ", ".join([str(interval) for interval in self._coverage_intervals]),
                )
            )

        # Compute the weights

        weights = np.empty_like(self._matrix_list, float)

        # These "effective intervals" are how much of the coverage interval is really used for each matrix
        # NOTE: the length of effective_intervals list *will not be* the same as the weight mask or the matrix_list.
        # There are as many effective intervals as matrices with weight > 0

        effective_intervals = []

        for i, matrix in enumerate(self._matrix_list):

            if matrices_mask[i]:

                # A matrix of interest
                this_coverage_interval = self._coverage_intervals[i]

                # See how much it overlaps with the interval of interest
                this_effective_interval = this_coverage_interval.intersect(
                    interval_of_interest
                )

                effective_intervals.append(this_effective_interval)

                # Now compute the weight

                if switch == "counts":

                    # Weight according to the number of events
                    weights[i] = self._counts_getter(
                        this_effective_interval.start_time,
                        this_effective_interval.stop_time,
                    )

                elif switch == "exposure":

                    # Weight according to the exposure
                    weights[i] = self._exposure_getter(
                        this_effective_interval.start_time,
                        this_effective_interval.stop_time,
                    )

            else:

                # Uninteresting matrix
                weights[i] = 0.0

        # if all weights are zero, there is something clearly wrong with the exposure or the counts computation
        assert (
            np.sum(weights) > 0
        ), "All weights are zero. There must be a bug in the exposure or counts computation"

        # Check that the first matrix with weight > 0 has an effective interval starting at the beginning of
        # the interval of interest (otherwise it means that part of the interval of interest is not covered!)

        if effective_intervals[0].start_time != interval_of_interest.start_time:

            raise IntervalOfInterestNotCovered(
                "The interval of interest (%s) is not covered by %s"
                % (interval_of_interest, effective_intervals[0])
            )

        # Check that the last matrix with weight > 0 has an effective interval starting at the beginning of
        # the interval of interest (otherwise it means that part of the interval of interest is not covered!)

        if effective_intervals[-1].stop_time != interval_of_interest.stop_time:
            raise IntervalOfInterestNotCovered(
                "The interval of interest (%s) is not covered by %s"
                % (interval_of_interest, effective_intervals[0])
            )

        # Lastly, check that there is no interruption in coverage (bad time intervals are *not* supported)
        all_tstarts = np.array([x.start_time for x in effective_intervals])
        all_tstops = np.array([x.stop_time for x in effective_intervals])

        if not np.all((all_tstops[:-1] == all_tstarts[1:])):

            raise GapInCoverageIntervals(
                "Gap in coverage! Bad time intervals are not supported!"
            )

        return weights

    @property
    def ebounds(self):

        return self._matrix_list[0].ebounds

    @property
    def monte_carlo_energies(self):

        return self._matrix_list[0].monte_carlo_energies


####################################################################################
# The following classes are used to create OGIP-compliant response files
# (at the moment only RMF are supported)


class EBOUNDS(FITSExtension):

    _HEADER_KEYWORDS = (
        ("EXTNAME", "EBOUNDS", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("CHANTYPE", "PI", "Channel type"),
        ("CONTENT", "OGIPResponse Matrix", "File content"),
        ("HDUCLAS1", "RESPONSE", "Extension contains response data  "),
        ("HDUCLAS2", "EBOUNDS ", "Extension contains EBOUNDS"),
        ("TLMIN1", 1, "Minimum legal channel number"),
    )

    def __init__(self, energy_boundaries):
        """
        Represents the EBOUNDS extension of a response matrix FITS file

        :param energy_boundaries: lower bound of channel energies (in keV)
        """

        n_channels = len(energy_boundaries) - 1

        data_tuple = (
            ("CHANNEL", list(range(1, n_channels + 1))),
            ("E_MIN", energy_boundaries[:-1] * u.keV),
            ("E_MAX", energy_boundaries[1:] * u.keV),
        )

        super(EBOUNDS, self).__init__(data_tuple, self._HEADER_KEYWORDS)


class MATRIX(FITSExtension):
    """
    Represents the MATRIX extension of a response FITS file following the OGIP format

    :param mc_energies_lo: lower bound of MC energies (in keV)
    :param mc_energies_hi: hi bound of MC energies (in keV)
    :param channel_energies_lo: lower bound of channel energies (in keV)
    :param channel_energies_hi: hi bound of channel energies (in keV
    :param matrix: the redistribution matrix, representing energy dispersion effects
    """

    _HEADER_KEYWORDS = [
        ("EXTNAME", "MATRIX", "Extension name"),
        ("HDUCLASS", "OGIP    ", "format conforms to OGIP standard"),
        ("HDUVERS", "1.1.0   ", "Version of format (OGIP memo CAL/GEN/92-002a)"),
        (
            "HDUDOC",
            "OGIP memos CAL/GEN/92-002 & 92-002a",
            "Documents describing the forma",
        ),
        ("HDUVERS1", "1.0.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUVERS2", "1.1.0   ", "Obsolete - included for backwards compatibility"),
        ("HDUCLAS1", "RESPONSE", "dataset relates to spectral response"),
        ("HDUCLAS2", "RSP_MATRIX", "dataset is a spectral response matrix"),
        ("HDUCLAS3", "REDIST", "dataset represents energy dispersion only"),
        ("CHANTYPE", "PI ", "Detector Channel Type in use (PHA or PI)"),
        ("DETCHANS", None, "Number of channels"),
        ("FILTER", "", "Filter used"),
        ("TLMIN4", 1, "Minimum legal channel number"),
    ]

    def __init__(self, mc_energies, channel_energies, matrix):

        n_mc_channels = len(mc_energies) - 1
        n_channels = len(channel_energies) - 1

        assert matrix.shape == (
            n_channels,
            n_mc_channels,
        ), "Matrix has the wrong shape. Should be %i x %i, got %i x %i" % (
            n_channels,
            n_mc_channels,
            matrix.shape[0],
            matrix.shape[1],
        )

        ones = np.ones(n_mc_channels, np.int16)

        # We need to format the matrix as a list of n_mc_channels rows of n_channels length

        data_tuple = (
            ("ENERG_LO", mc_energies[:-1] * u.keV),
            ("ENERG_HI", mc_energies[1:] * u.keV),
            ("N_GRP", ones),
            ("F_CHAN", ones),
            ("N_CHAN", np.ones(n_mc_channels, np.int16) * n_channels),
            ("MATRIX", matrix.T),
        )

        super(MATRIX, self).__init__(data_tuple, self._HEADER_KEYWORDS)

        # Update DETCHANS
        self.hdu.header.set("DETCHANS", n_channels)


class SPECRESP_MATRIX(MATRIX):
    """
    Represents the SPECRESP_MATRIX extension of a response FITS file following the OGIP format

    :param mc_energies_lo: lower bound of MC energies (in keV)
    :param mc_energies_hi: hi bound of MC energies (in keV)
    :param channel_energies_lo: lower bound of channel energies (in keV)
    :param channel_energies_hi: hi bound of channel energies (in keV
    :param matrix: the redistribution matrix, representing energy dispersion effects and effective area information
    """

    def __init__(self, mc_energies, channel_energies, matrix):

        # This is essentially exactly the same as MATRIX, but with a different extension name

        super(SPECRESP_MATRIX, self).__init__(mc_energies, channel_energies, matrix)

        # Change the extension name
        self.hdu.header.set("EXTNAME", "SPECRESP MATRIX")
        self.hdu.header.set("HDUCLAS3", "FULL")


class RMF(FITSFile):
    """
    A RMF file, the OGIP format for a matrix representing energy dispersion effects.

    """

    def __init__(self, mc_energies, ebounds, matrix, telescope_name, instrument_name):

        # Make sure that the provided iterables are of the right type for the FITS format

        mc_energies = np.array(mc_energies, np.float32)

        ebounds = np.array(ebounds, np.float32)

        # Create EBOUNDS extension
        ebounds_ext = EBOUNDS(ebounds)

        # Create MATRIX extension
        matrix_ext = MATRIX(mc_energies, ebounds, matrix)

        # Set telescope and instrument name
        matrix.hdu.header.set("TELESCOP", telescope_name)
        matrix.hdu.header.set("INSTRUME", instrument_name)

        # Create FITS file
        super(RMF, self).__init__(fits_extensions=[ebounds_ext, matrix_ext])


class RSP(FITSFile):
    """
    A response file, the OGIP format for a matrix representing both energy dispersion effects and effective area,
    in the same matrix.

    """

    def __init__(self, mc_energies, ebounds, matrix, telescope_name, instrument_name):

        # Make sure that the provided iterables are of the right type for the FITS format

        mc_energies = np.array(mc_energies, np.float32)

        ebounds = np.array(ebounds, np.float32)

        # Create EBOUNDS extension
        ebounds_ext = EBOUNDS(ebounds)

        # Create MATRIX extension
        matrix_ext = SPECRESP_MATRIX(mc_energies, ebounds, matrix)

        # Set telescope and instrument name
        matrix_ext.hdu.header.set("TELESCOP", telescope_name)
        matrix_ext.hdu.header.set("INSTRUME", instrument_name)

        # Create FITS file
        super(RSP, self).__init__(fits_extensions=[ebounds_ext, matrix_ext])
