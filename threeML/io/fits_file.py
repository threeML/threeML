from builtins import str
from builtins import object
from astropy.io import fits
import numpy as np
import astropy.units as u
import pkg_resources
import six

from threeML.io.logging import setup_logger

log = setup_logger(__name__)

# From https://heasarc.gsfc.nasa.gov/docs/software/fitsio/c/c_user/node20.html
# Codes for the data type of binary table columns and/or for the
# data type of variables when reading or writing keywords or data:
#                               DATATYPE               TFORM CODE
#   #define TBIT          1  /*                            'X' */
#   #define TBYTE        11  /* 8-bit unsigned byte,       'B' */
#   #define TLOGICAL     14  /* logicals (int for keywords     */
#                            /*  and char for table cols   'L' */
#   #define TSTRING      16  /* ASCII string,              'A' */
#   #define TSHORT       21  /* signed short,              'I' */
#   #define TLONG        41  /* signed long,                   */
#   #define TLONGLONG    81  /* 64-bit long signed integer 'K' */
#   #define TFLOAT       42  /* single precision float,    'E' */
#   #define TDOUBLE      82  /* double precision float,    'D' */
#   #define TCOMPLEX     83  /* complex (pair of floats)   'C' */
#   #define TDBLCOMPLEX 163  /* double complex (2 doubles) 'M' */
#   The following data type codes are also supported by CFITSIO:
#   #define TINT         31  /* int                            */
#   #define TSBYTE       12  /* 8-bit signed byte,         'S' */
#   #define TUINT        30  /* unsigned int               'V' */
#   #define TUSHORT      20  /* unsigned short             'U'  */
#   #define TULONG       40  /* unsigned long                  */
#   The following data type code is only for use with fits\_get\_coltype
#   #define TINT32BIT    41  /* signed 32-bit int,         'J' */


_NUMPY_TO_FITS_CODE = {
    # Integers
    np.int16: "I",
    np.int32: "J",
    np.int64: "K",
    np.uint16: "I",
    np.uint32: "J",
    # Floating point
    np.float32: "E",
    np.float64: "D",
}


class FITSFile(object):
    def __init__(self, primary_hdu=None, fits_extensions=None):

        hdu_list = []

        if primary_hdu is None:

            primary_hdu = fits.PrimaryHDU()

        else:

            assert isinstance(primary_hdu, fits.PrimaryHDU)

        hdu_list.append(primary_hdu)

        if fits_extensions is not None:

            fits_extensions = list(fits_extensions)

            hdu_list.extend([x.hdu for x in fits_extensions])

        # We embed instead of subclassing because the HDUList class has some weird interaction with the
        # __init__ and __new__ methods which makes difficult to do so (we couldn't figure it out)

        self._hdu_list = fits.HDUList(hdus=hdu_list)

    def writeto(self, *args, **kwargs):

        self._hdu_list.writeto(*args, **kwargs)

    # Update the docstring to be the same as the method we are wrapping

    writeto.__doc__ = fits.HDUList.writeto.__doc__

    def __getitem__(self, item):

        return self._hdu_list.__getitem__(item)

    def info(self, output=None):

        self._hdu_list.info(output)

    info.__doc__ = fits.HDUList.info.__doc__

    def index_of(self, key):

        return self._hdu_list.index_of(key)

    index_of.__doc__ = fits.HDUList.index_of.__doc__


class FITSExtension(object):

    # I use __new__ instead of __init__ because I need to use the classmethod .from_columns instead of the
    # constructor of fits.BinTableHDU

    def __init__(self, data_tuple, header_tuple):

        # Generate the header from the dictionary

        header = fits.Header(header_tuple)

        # Loop over the columns and generate them
        fits_columns = []

        for column_name, column_data in data_tuple:

            # Get type of column
            # NOTE: we assume the type is the same for the entire column

            test_value = column_data[0]

            # Generate FITS column

            # By default a column does not have units, unless the content is an astropy.Quantity

            units = None

            if isinstance(test_value, u.Quantity):

                # Probe the format

                try:

                    # Use the one already defined, if possible

                    format = _NUMPY_TO_FITS_CODE[column_data.dtype.type]

                except AttributeError:

                    # Try to infer it. Note that this could unwillingly upscale a float16 to a float32, for example

                    format = _NUMPY_TO_FITS_CODE[np.array(test_value.value).dtype.type]

                # check if this is a vector of quantities

                if test_value.shape:

                    format = "%i%s" % (test_value.shape[0], format)

                # Store the unit as text

                units = str(test_value.unit)

            elif isinstance(test_value, six.string_types):

                # Get maximum length, but make 1 as minimum length so if the column is completely made up of empty
                # string we still can work

                max_string_length = max(len(max(column_data, key=len)), 1)

                format = "%iA" % max_string_length

            elif np.isscalar(test_value):

                format = _NUMPY_TO_FITS_CODE[np.array(test_value).dtype.type]

            elif isinstance(test_value, list) or isinstance(test_value, np.ndarray):

                # Probably a column array
                # Check that we can convert it to a proper numpy type

                try:

                    # Get type of first number

                    col_type = np.array(test_value[0]).dtype.type

                except:

                    raise RuntimeError(
                        "Could not understand type of column %s" % column_name
                    )

                # Make sure we are not dealing with objects
                assert col_type != np.object and col_type != np.object_

                try:

                    _ = np.array(test_value, col_type)

                except:

                    raise RuntimeError(
                        "Column %s contain data which cannot be coerced to %s"
                        % (column_name, col_type)
                    )

                else:

                    # see if it is a string array

                    if test_value.dtype.type == np.string_:

                        max_string_length = max(column_data, key=len).dtype.itemsize

                        format = "%iA" % max_string_length

                    else:

                        # All good. Check the length
                        # NOTE: variable length arrays are not supported
                        line_length = len(test_value)
                        format = "%i%s" % (line_length, _NUMPY_TO_FITS_CODE[col_type])

            else:

                # Something we do not know

                raise RuntimeError(
                    "Column %s in dataframe contains objects which are not strings"
                    % column_name
                )

            this_column = fits.Column(
                name=column_name, format=format, unit=units, array=column_data
            )

            fits_columns.append(this_column)

        # Create the extension

        self._hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs(fits_columns), header=header
        )

        # update the header to indicate that the file was created by 3ML
        self._hdu.header.set(
            "CREATOR",
            "3ML v.%s" % (pkg_resources.get_distribution("threeML").version),
            "(G.Vianello, giacomov@slac.stanford.edu)",
        )

    @property
    def hdu(self):

        return self._hdu

    @classmethod
    def from_fits_file_extension(cls, fits_extension):

        data = fits_extension.data

        data_tuple = []

        for name in data.columns.names:

            data_tuple.append((name, data[name]))

        header_tuple = list(fits_extension.header.items())

        return cls(data_tuple, header_tuple)
