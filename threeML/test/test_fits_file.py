from threeML.io.fits_file import FITSExtension, FITSFile
import numpy as np
import astropy.io.fits as fits

import pytest


class DUMMYEXT(FITSExtension):
    def __init__(self, test_value):

        data_list = [("TEST_VALUE", test_value)]

        super(DUMMYEXT, self).__init__(
            tuple(data_list), (("EXTNAME", "TEST", "Extension name"),)
        )


class DUMMYFITS(FITSFile):
    def __init__(self, test_value):

        dummy_extension = DUMMYEXT(test_value)

        super(DUMMYFITS, self).__init__(fits_extensions=[dummy_extension])


def test_fits_file():

    dtypes = [
        np.int16,
        np.int32,
        np.int64,
        np.uint16,
        np.uint32,
        np.float32,
        np.float64,
    ]
    dtype_keys = ["I", "J", "K", "I", "J", "E", "D"]

    for i, dt in enumerate(dtypes):

        test_values = np.ones(10, dtype=dt)

        dummy_fits = DUMMYFITS(test_value=test_values)

        assert len(dummy_fits._hdu_list) == 2

        assert dummy_fits.index_of("TEST") == 1

        assert dummy_fits["TEST"].header["TFORM1"] == dtype_keys[i]

        assert np.alltrue(dummy_fits["TEST"].data["TEST_VALUE"] == test_values)

        file_name = "test_fits%d.fits" % i

        dummy_fits.writeto(file_name, overwrite=True)

        with pytest.raises(IOError):

            dummy_fits.writeto(file_name, overwrite=False)

        read_dummy_fits = fits.open(file_name)

        assert len(read_dummy_fits) == 2

        assert read_dummy_fits.index_of("TEST") == 1

        assert read_dummy_fits["TEST"].header["TFORM1"] == dtype_keys[i]

        assert np.alltrue(read_dummy_fits["TEST"].data["TEST_VALUE"] == test_values)
