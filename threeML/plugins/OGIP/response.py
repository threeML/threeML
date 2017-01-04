import astropy.io.fits as pyfits
import numpy as np
import warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re


class GenericResponse(object):
    def __init__(self, matrix, ebounds, mc_channels, rsp_file=None, arf_file=None):
        """

        Generic response class that accepts a full matrix, ebounds in vstack form
        the mc channels in vstack form, an options rsp file and arf file.

        Therefore, an OGIP style RSP from a file is not required if the matrix,
        ebounds, and mc channels exist.


        :param matrix: an N_CHANS X N_ENERGIES response matrix
        :param ebounds: the energy bounds of the channles
        :param mc_channels: the energy channels
        :param rsp_file: file the rsp was possibly read from
        :param arf_file: an optional arf file
        """

        # we simply store all the variables to the class

        self._matrix = matrix

        self._ebounds = ebounds

        self._mc_channels = mc_channels

        self._rsp_file = rsp_file

        self._arf_file = arf_file

        self._differential_function = None

        self._integral_function = None

        # Store the name of the file
        self._rsp_file = rsp_file
        self._arf_file = arf_file

    @property
    def rsp_filename(self):
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

    @property
    def first_channel(self):
        """
        The first channel of the channel array. Correpsonds to
        TLMIN keyword in FITS files

        :return: first channel
        """
        return int(self._first_channel)

    def _read_ebounds(self, ebounds_extension):
        """
        reads the ebounds from an OGIP response

        :param ebounds_extension: an RSP ebounds extension
        :return:
        """

        ebounds = np.vstack([ebounds_extension.data.field("E_MIN"),
                             ebounds_extension.data.field("E_MAX")]).T

        ebounds = ebounds.astype(float)

        return ebounds

    def _read_mc_channels(self, data):
        """
        reads the mc_channels from an OGIP response

        :param data: data from a RSP MATRIX
        :return:
        """

        mc_channels = np.vstack([data.field("ENERG_LO"),
                                 data.field("ENERG_HI")]).T

        mc_channels = mc_channels.astype(float)

        return mc_channels

    def _read_matrix(self, data, header, column_name='MATRIX'):

        n_channels = header.get("DETCHANS")

        assert n_channels is not None, "Matrix is improperly formatted. No DETCHANS keyword."

        # The header contains a keyword which tells us the first legal channel. It is TLMIN of the F_CHAN column
        # NOTE: TLMIN keywords start at 1, so TLMIN1 is the minimum legal value for the first column. So we need
        # to add a +1 because of course the numbering of lists (data.columns.names) starts at 0

        f_chan_column_pos = data.columns.names.index("F_CHAN") + 1

        try:
            tlmin_fchan = header["TLMIN%i" % f_chan_column_pos]

        except(KeyError):
            warnings.warn('No TLMIN keyword found. This DRM does not follow OGIP standards. Assuming TLMIN=1')
            tlmin_fchan = 1

        # Store the first channel as a property
        self._first_channel = tlmin_fchan

        rsp = np.zeros([data.shape[0], n_channels], float)

        n_grp = data.field("N_GRP")

        # The numbering of channels could start at 0, or at some other number (usually 1). Of course the indexing
        # of arrays starts at 0. So let's offset the F_CHAN column to account for that

        f_chan = data.field("F_CHAN") - tlmin_fchan
        n_chan = data.field("N_CHAN")

        # In certain matrices where compression has not been used, n_grp, f_chan and n_chan are not array columns,
        # but simple scalars. Expand then their dimensions so that we don't need to customize the code below

        if n_grp.ndim == 1:
            n_grp = np.expand_dims(n_grp, 1)

        if f_chan.ndim == 1:
            f_chan = np.expand_dims(f_chan, 1)

        if n_chan.ndim == 1:
            n_chan = np.expand_dims(n_chan, 1)

        matrix = data.field(column_name)

        for i, row in enumerate(data):

            m_start = 0

            for j in range(n_grp[i]):
                rsp[i, f_chan[i][j]: f_chan[i][j] + n_chan[i][j]] = matrix[i][m_start:m_start + n_chan[i][j]]

                m_start += n_chan[i][j]

        return rsp.T

    def _read_arf_file(self, arf_file, current_matrix, current_mc_channels):
        """
        read an arf file and apply it to the current_matrix

        :param arf_file:
        :param current_matrix:
        :param current_mc_channels:
        :return:
        """

        arf_file = sanitize_filename(arf_file)

        assert file_existing_and_readable(arf_file.split("{")[0]), "Ancillary file %s not existing or not " \
                                                                   "readable" % arf_file

        with pyfits.open(arf_file) as f:

            data = f['SPECRESP'].data

        arf = data.field('SPECRESP')

        # Check that arf and rmf have same dimensions

        if arf.shape[0] != current_matrix.shape[1]:
            raise IOError("The ARF and the RMF file does not have the same number of channels")

        # Check that the ENERG_LO and ENERG_HI for the RMF and the ARF
        # are the same

        arf_mc_channels = np.vstack([data.field("ENERG_LO"),
                                     data.field("ENERG_HI")]).T

        # Declare the mc channels different if they differ by more than
        # 1%

        idx = (current_mc_channels > 0)

        diff = (current_mc_channels[idx] - arf_mc_channels[idx]) / current_mc_channels[idx]

        if diff.max() > 0.01:
            raise IOError("The ARF and the RMF have one or more MC channels which differ by more than 1%")

        # Multiply ARF and RMF

        matrix = current_matrix * arf

        return matrix

    @property
    def ebounds(self):
        """

        Returns the ebounds of the RSP.

        :return:
        """
        return self._ebounds

    def set_function(self, differentialFunction, integralFunction=None):
        """
        Set the function to be used for the convolution
        """

        self._differential_function = differentialFunction

        self._integral_function = integralFunction

    def convolve(self):

        true_fluxes = self._integral_function(self._mc_channels[:, 0],
                                              self._mc_channels[:, 1])

        # Sometimes some channels have 0 lenths, or maybe they start at 0, where
        # many functions (like a power law) are not defined. In the response these
        # channels have usually a 0, but unfortunately for a computer
        # inf * zero != zero. Thus, let's force this. We avoid checking this situation
        # in details because this would have a HUGE hit on performances

        idx = np.isfinite(true_fluxes)
        true_fluxes[~idx] = 0

        folded_counts = np.dot(true_fluxes, self._matrix.T)

        return folded_counts

    def energy_to_channel(self, energy):

        '''Finds the channel containing the provided energy.
        NOTE: returns the channel index (starting at zero),
        not the channel number (likely starting from 1)'''

        # Get the index of the first ebounds upper bound larger than energy

        try:

            idx = next(idx for idx,
                               value in enumerate(self._ebounds[:, 1])
                       if value >= energy)

        except StopIteration:

            # No values above the given energy, return the last channel
            return self._ebounds[:, 1].shape[0]

        return idx

    def plot_matrix(self):

        fig, ax = plt.subplots()

        image = self._matrix.T

        ax.set_xscale('log')
        ax.set_yscale('log')

        idx1 = 0
        idx2 = 0

        # Some times the lower edges may be zero, so we skip them

        if self._mc_channels[0, 0] == 0:
            idx1 = 1

        if self._ebounds[0, 0] == 0:
            idx2 = 1

        ax.imshow(image[idx1:, idx2:], extent=(self._mc_channels[idx1, 0],
                                               self._mc_channels[-1, 1],
                                               self._ebounds[idx2, 0],
                                               self._ebounds[-1, 1]),
                  aspect=1.5,
                  cmap=cm.BrBG_r)

        # if show_energy is not None:
        #    ener_val = Quantity(show_energy).to(self.reco_energy.unit).value
        #    ax.hlines(ener_val, 0, 200200, linestyles='dashed')

        ax.set_ylabel('True energy (keV)')
        ax.set_xlabel('Reco energy (keV)')

        return fig


class Response(GenericResponse):
    def __init__(self, rsp_file, arf_file=None):
        """

        :param rsp_file:
        :param arf_file:
        """

        # Now make sure that the response file exist

        rsp_file = sanitize_filename(rsp_file)

        assert file_existing_and_readable(rsp_file.split("{")[0]), "Response file %s not existing or not " \
                                                                   "readable" % rsp_file

        # Check if we are dealing with a .rsp2 file (containing more than
        # one response). This is checked by looking for the syntax
        # [responseFile]{[responseNumber]}

        if '{' in rsp_file:

            tokens = rsp_file.split("{")
            rsp_file = tokens[0]
            rsp_number = int(tokens[-1].split('}')[0].replace(" ", ""))

        else:

            rsp_number = 1

        # Read the response
        with pyfits.open(rsp_file) as f:

            try:

                # This is usually when the response file contains only the energy dispersion

                data = f['MATRIX', rsp_number].data
                header = f['MATRIX', rsp_number].header

                if arf_file is None:
                    warnings.warn("The response is in an extension called MATRIX, which usually means you also "
                                  "need an ancillary file (ARF) which you didn't provide. You should refer to the "
                                  "documentation  of the instrument and make sure you don't need an ARF.")

            except:

                # Other detectors might use the SPECRESP MATRIX name instead, usually when the response has been
                # already convoluted with the effective area

                # Note that here we are not catching any exception, because
                # we have to fail if we cannot read the matrix

                data = f['SPECRESP MATRIX', rsp_number].data
                header = f['SPECRESP MATRIX', rsp_number].header

            # Sometimes .rsp files contains a weird format featuring variable-length
            # arrays. Historically those confuse pyfits quite a lot, so we ensure
            # to transform them into standard numpy matrices to avoid issues

            matrix = self._read_matrix(data, header)

            ebounds = self._read_ebounds(f['EBOUNDS'])

            mc_channels = self._read_mc_channels(data)

            # Now let's see if we have a ARF, if yes, read it

            if arf_file is not None and (arf_file.upper() != "NONE"):
                matrix = self._read_arf_file(arf_file, matrix, mc_channels)

        super(Response, self).__init__(matrix=matrix,
                                       ebounds=ebounds,
                                       mc_channels=mc_channels,
                                       rsp_file=rsp_file,
                                       arf_file=arf_file)


class WeightedResponse(GenericResponse):
    def _init_(self, rsp_file, trigger_time, count_getter, exposure_getter, arf_file=None):
        """
        A weighted response function that recalculates the response from a series of
        responses based of the time intervals to be analyzed. Supports multiple, disjoint
        time intervals.

        The class is initialized, but no response is generated until rsp.set_interval(*intervals)
        is called.

        Currently, ARFs are not supported.


        :param rsp_file: the RSP2 style file
        :param trigger_time: the trigger time of the event
        :param count_getter: a function to get counts between intervals: f(tmin,tmax)
        :param exposure_getter: a function to get exposure between intervals: f(tmin,tmax)
        :param arf_file: optional arf (not supported yet!)
        :return:
        """

        # lock the count and exposure functions to the
        # object

        self._count_getter = count_getter
        self._exposure_getter = exposure_getter

        # make the rsp file proper
        rsp_file = sanitize_filename(rsp_file)

        # really, there should be no braces
        assert file_existing_and_readable(rsp_file.split("{")[0]), "Response file %s not existing or not " \
                                                                   "readable" % rsp_file
        # lock the trigger time
        self._trigger_time = trigger_time

        # Read the response
        with pyfits.open(rsp_file) as f:

            self._n_responses = f['PRIMARY'].header['DRM_NUM']
            self._matrix_start = np.zeros(self._n_responses)
            self._matrix_stop = np.zeros(self._n_responses)

            matrices = []

            # try either option of matrix

            try:

                # we will read all the matrices and save them

                for rsp_number in range(1, self._n_responses + 1):
                    # This is usually when the response file contains only the energy dispersion

                    data = f['MATRIX', rsp_number].data
                    header = f['MATRIX', rsp_number].header

                    matrix = self._read_matrix(data, header)

                    matrices.append(matrix)

                    # Find the start and stop time of the interval covered
                    # by this response matrix
                    header_start = header["TSTART"]
                    header_stop = header["TSTOP"]

                    self._matrix_start[rsp_number - 1] = header_start - trigger_time
                    self._matrix_stop[rsp_number - 1] = header_stop - trigger_time

                if arf_file is None:
                    warnings.warn("The response is in an extension called MATRIX, which usually means you also "
                                  "need an ancillary file (ARF) which you didn't provide. You should refer to the "
                                  "documentation  of the instrument and make sure you don't need an ARF.")

            except:

                # Other detectors might use the SPECRESP MATRIX name instead, usually when the response has been
                # already convoluted with the effective area

                # Note that here we are not catching any exception, because
                # we have to fail if we cannot read the matrix


                # we will read all the matrices and save them
                for rsp_number in range(1, self._n_responses + 1):
                    # This is usually when the response file contains only the energy dispersion

                    data = f['SPECRESP MATRIX', rsp_number].data
                    header = f['SPECRESP MATRIX', rsp_number].header

                    matrix = self._read_matrix(data, header)

                    matrices.append(matrix)

                    # Find the start and stop time of the interval covered
                    # by this response matrix
                    header_start = header["TSTART"]
                    header_stop = header["TSTOP"]

                    self._matrix_start[rsp_number - 1] = header_start - trigger_time
                    self._matrix_stop[rsp_number - 1] = header_stop - trigger_time

            # read the ebounds and mc channels

            ebounds = self._read_ebounds(f['EBOUNDS'])

            mc_channels = self._read_mc_channels(data)

            # currently, a weighted ARF is not supported

            if arf_file is not None and (arf_file.upper() != "NONE"):
                raise NotImplementedError('WeightedResponse does not yet support ARFs')

                # matrix = self._read_arf_file(arf_file, matrix, mc_channels)

            self._matrices = np.array(matrices)

            # reshape the matrix to be N_RSP X N_CHANS X N_ENRG
            self._matrices.reshape((self._n_responses, matrices[0].shape[0], matrices[0].shape[1]))

            # we are not going to call the GeneralResponse constructor just yet
            # the weighted RSP is generated when a time section is made.
            # Therefore, we will save the ebounds, mc_channels, rsp_file
            # and arf_file to the object and pass them (redundantly) to
            # the constructor only when a selection is made

            self._ebounds = ebounds
            self._mc_channels = mc_channels
            self._rsp_file = rsp_file
            self._arf_file = arf_file

    def _weight_response(self):
        """
        The matrix cover the period going from the middle point between its start
        time and the start time of the previous matrix (or its start time if it is the
        first matrix of the file), and the middle point between its start time and its
        stop time (that is equal to the start time of the next one):

                  tstart                             tstop
                    |==================================|
        |--------------x---------------|-----------x-----------|----------x--..
        rspStart1    rspStop1=    headerStart2  rspStop2= headerStart3  rspStop3
                    rspStart2                  rspStart3

        covered by: |m1 |           m2             | m3|
        """

        # we use a list of bools to select matrices that need to be summed
        matrices_to_use = np.zeros(self._n_responses, dtype=bool)

        # lists of the true starts and stops
        # for all matrices used

        all_rsp_stops = []
        all_rsp_starts = []

        # loop through all the intervals selected

        for start, stop in zip(self._tstarts, self._tstops):

            # initialize to nothing
            previous_rsp_stop = None

            # we need to find the first response for EACH time interval
            is_first_response = True

            # now we loop through the matrices
            # and log which ones are need and
            # what their boundaries are

            for rsp_start, rsp_stop, rsp_number in zip(self._matrix_start, self._matrix_stop, range(self._n_responses)):

                if is_first_response:

                    # this first  matrix

                    if rsp_number == self._n_responses - 1:

                        # this is the first and last matrix

                        this_rsp_start = rsp_start

                        this_rsp_stop = rsp_stop

                    else:

                        this_rsp_start = rsp_start
                        this_rsp_stop = 0.5 * (rsp_start + rsp_stop)  # midpoint

                        is_first_response = False

                elif rsp_number == self._n_responses - 1:

                    # this is the last matrix

                    this_rsp_start = previous_rsp_stop
                    this_rsp_stop = rsp_stop

                else:
                    # we have more to go

                    this_rsp_start = previous_rsp_stop
                    this_rsp_stop = 0.5 * (rsp_start + rsp_stop)  # midpoint

                # log where we stopped on this iteration

                previous_rsp_stop = this_rsp_stop

                if (start <= this_rsp_stop and this_rsp_start <= stop):

                    # Found a matrix covering a part of the interval:
                    # adding it to the list

                    matrices_to_use[rsp_number] = True

                    # Get the "true" start time of the time sub-interval covered by this matrix

                    true_rsp_start = max(this_rsp_start, start)

                    # Get the "true" stop time of the time sub-interval covered by this matrix

                    if rsp_number == self._n_responses - 1:

                        # Since there are no matrices after this one, this has to cover until the end of the interval

                        if this_rsp_stop < stop:

                            # the matrix interval has ended before the required interval
                            # so we are going to extend the validity of the matrix interval
                            # out to the end of the interval

                            true_rsp_stop = stop

                        else:
                            # should we instead use the this_rsp_stop... need to ask
                            true_rsp_stop = stop  #### Seems there is a bug here

                    else:

                        true_rsp_stop = min(this_rsp_stop, stop)

                    # Ok, save also the boundaries this round

                    all_rsp_starts.append(true_rsp_start)

                    all_rsp_stops.append(true_rsp_stop)

                if stop <= this_rsp_stop:
                    # we're done with this interval
                    # so lets save time

                    break

        # Now we have logged all the matrices and boundaries needed for EACH interval
        # we need to figure out the weighting. Since multiple intervals are possible,
        # we use the total counts and exposure over all intervals rather than the individual
        # intervals as was done previously. We also use the exposure, rather than just the interval
        # so that dead time is accounted for.

        # initialize the weighting

        weight = []

        n_summable_matrices = matrices_to_use.sum()

        assert n_summable_matrices > 0, "There were no matrices in the interval requested. This is a bug" # pragma: no cover

        if n_summable_matrices > 1:

            assert n_summable_matrices == len(
                all_rsp_starts), 'This is a bug. We have %d matrices to use but only %d intervals found' % (
            n_summable_matrices, len(all_rsp_starts)) # pragma: no cover

            # we have more than one matrix

            if self._total_counts_this_selection <= 0:

                # we will weight by exposure instead

                if sum(weight) == 0:

                    for idx, matrix in enumerate(self._matrices[matrices_to_use]):
                        rsp_interval_exposure = self._exposure_getter(all_rsp_stops[idx], all_rsp_starts[idx])

                        this_weight = rsp_interval_exposure / self._total_exposure_this_selection

                        weight.append(this_weight)

            else:


                for idx, matrix in enumerate(self._matrices[matrices_to_use]):

                    this_rsp_start = all_rsp_starts[idx]

                    this_rsp_stop = all_rsp_stops[idx]

                    # find out how many counts we have in the RSP boundaries

                    rsp_interval_counts = self._count_getter(this_rsp_start, this_rsp_stop)

                    if rsp_interval_counts > 0:

                        this_weight = float(rsp_interval_counts) / float(self._total_counts_this_selection)

                    else:

                        this_weight = 0.

                    weight.append(this_weight)

            if sum(weight) != 1.:

                # we will need to redistribute the weight over the RSP based off exposure

                leftover_weight = 1. - sum(weight)

                for idx, matrix in enumerate(self._matrices[matrices_to_use]):

                    rsp_interval_exposure = self._exposure_getter(all_rsp_stops[idx], all_rsp_starts[idx])

                    this_exposure_weight = rsp_interval_exposure / self._total_exposure_this_selection

                    weight[idx] += this_exposure_weight * leftover_weight


        else:

            # we only have one matrix

            weight = [1]

        weight = np.array(weight)

        assert weight.shape[0] == self._matrices[matrices_to_use].shape[
            0], "The weights disagree with the matrices... this is a bug"


        # save these variables for plotting information about the weighting

        self._true_rsp_intervals = np.vstack((all_rsp_starts,all_rsp_stops))
        self._weight = weight
        self._use_matrices = matrices_to_use

        weighted_matrix = np.multiply(weight, self._matrices[matrices_to_use]).sum()

        return weighted_matrix

    @staticmethod
    def _parse_time_interval(time_interval):
        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    def set_time_interval(self, *intervals):

        self._tstarts = []
        self._tstops = []

        # intialize the total counts and exposure over all intervals
        self._total_counts_this_selection = 0
        self._total_exposure_this_selection = 0

        # build a list of intervals

        for interval in intervals:
            tmin, tmax = self._parse_time_interval(interval)

            self._tstarts.append(tmin)
            self._tstops.append(tmax)

            # add up the counts and exposure
            self._total_counts_this_selection += self._count_getter(tmin, tmax)
            self._total_exposure_this_selection += self._exposure_getter(tmin, tmax)

        # now we can weight the matrix
        weighted_matrix = self._weight_response()

        # call the constructor

        super(WeightedResponse, self).__init__(matrix=weighted_matrix,
                                               ebounds=self._ebounds,
                                               mc_channels=self._mc_channels,
                                               rsp_file=self._rsp_file,
                                               arf_file=self._arf_file)

    def display_response_weighting(self):

        fig, ax = plt.subplot()


        # plot the time intervals

        ax.hlines(min(self._weight),self._tstarts,self._tstops,color='red',label='selected intervals')
        ax.hlines(np.median(self._weight), self._true_rsp_intervals[0], self._true_rsp_intervals[1], color='green', label='true rsp intervals')
        ax.hlines(max(self._weight), self._matrix_start, self._matrix_stop, color='blue', label='rsp header intervals')

        mean_true_rsp_time = np.mean(self._true_rsp_intervals.T,axis=1)

        ax.plot(mean_true_rsp_time,self._weight,'+k', label='weight')


