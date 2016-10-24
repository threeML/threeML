import astropy.io.fits as pyfits
import numpy as np
import warnings
from threeML.io.file_utils import file_existing_and_readable, sanitize_filename
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class Response(object):

    def __init__(self, rsp_file, arf_file=None):

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

            self._matrix = self._read_matrix(data, header)

            self._ebounds = np.vstack([f['EBOUNDS'].data.field("E_MIN"),
                                       f['EBOUNDS'].data.field("E_MAX")]).T

            self._ebounds = self._ebounds.astype(float)



            self._mc_channels = np.vstack([data.field("ENERG_LO"),
                                           data.field("ENERG_HI")]).T

            self._mc_channels = self._mc_channels.astype(float)

            # Now let's see if we have a ARF, if yes, read it

            if arf_file is not None and (arf_file.upper() != "NONE"):

                arf_file = sanitize_filename(arf_file)

                assert file_existing_and_readable(arf_file.split("{")[0]), "Ancillary file %s not existing or not " \
                                                                           "readable" % arf_file

                with pyfits.open(arf_file) as f:

                    data = f['SPECRESP'].data

                arf = data.field('SPECRESP')

                # Check that arf and rmf have same dimensions

                if arf.shape[0] != self._matrix.shape[1]:

                    raise IOError("The ARF and the RMF file does not have the same number of channels")

                # Check that the ENERG_LO and ENERG_HI for the RMF and the ARF
                # are the same

                arf_mc_channels = np.vstack([data.field("ENERG_LO"),
                                             data.field("ENERG_HI")]).T

                # Declare the mc channels different if they differ by more than
                # 1%

                idx = (self._mc_channels > 0)

                diff = (self._mc_channels[idx] - arf_mc_channels[idx]) / self._mc_channels[idx]

                if diff.max() > 0.01:
                    raise IOError("The ARF and the RMF have one or more MC channels which differ by more than 1%")

                # Multiply ARF and RMF

                self._matrix = self._matrix * arf

        # Init everything else to none
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
