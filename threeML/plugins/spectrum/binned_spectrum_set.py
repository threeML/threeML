from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum

import astropy.io.fits as fits

class BinnedSpectrumSet(object):

    def __init__(self, binned_spectrum_list, reference_time=0.0):
        """

        :param binned_spectrum_list:
        :param reference_time:
        """

        self._binned_spectrum_list = binned_spectrum_list
        self._reference_time = reference_time

    @property
    def reference_time(self):

        return self._reference_time

    def __getitem__(self, item):

        return self._binned_spectrum_list[item]

    def __len__(self):

        return len(self._binned_spectrum_list)

    @classmethod
    def from_pha2_fits(cls, pha2_file):

        with fits.open(pha2_file) as f:

            try:

                HDUidx = f.index_of("SPECTRUM")

            except:

                raise RuntimeError("The input file %s is not in PHA format" % (pha2_file))

            spectrum = f[HDUidx]
            data = spectrum.data

            if "COUNTS" in data.columns.names:

                has_rates = False
                data_column_name = "COUNTS"

            elif "RATE" in data.columns.names:

                has_rates = True
                data_column_name = "RATE"

            else:

                raise RuntimeError("This file does not contain a RATE nor a COUNTS column. "
                                   "This is not a valid PHA file")

                # Determine if this is a PHA I or PHA II
            if len(data.field(data_column_name).shape) == 2:

                num_spectra = data.field(data_column_name).shape[0]

            else:

                raise RuntimeError("This appears to be a PHA I and not PHA II file")



            list_of_binned_spectra =[ BinnedSpectrum.from_fits_file('%s{%d}'%(pha2_file,spectrum_number),
                                                                    file_type='observed') for spectrum_number in range(1, num_spectra+1)]




