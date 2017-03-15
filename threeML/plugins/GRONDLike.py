import numpy as np
import pandas as pd
import collections
import urllib2
import os

from threeML.plugins.PhotometryLike import PhotometryLike
from threeML.plugins.photometry.filter_set import FilterSet
from threeML.plugins.photometry.photometric_data import PhotometryData
from threeML.config.config import threeML_config
from threeML.io.file_utils import if_directory_not_existing_then_make, file_existing_and_readable

__instrument_name = "GROND 7-band photometric imager"

_grond_filter_names = ['g', 'r', 'i', 'z', 'J', 'H', 'K']
_grond_magnitude_systems_vega = ['abmag', 'abmag', 'abmag', 'abmag', 'vega mag', 'vega mag', 'vega mag']
_grond_magnitude_systems_ab = ['abmag', 'abmag', 'abmag', 'abmag', 'abmag', 'abmag', 'abmag']


class GRONDLike(PhotometryLike):
    def __init__(self,
                 name,
                 g=None,
                 g_err=None,
                 r=None,
                 r_err=None,
                 i=None,
                 i_err=None,
                 z=None,
                 z_err=None,
                 J=None,
                 J_err=None,
                 H=None,
                 H_err=None,
                 K=None,
                 K_err=None,
                 grond_transmission=None,
                 use_vega=True,
                 verbose=True):
        """

        :param name: the plugin name
        :param g: the g band magnitude
        :param g_err: the g band magnitude error
        :param r: the r band magnitude
        :param r_err: the r band magnitude error
        :param i: the i band magnitude
        :param i_err: the i band magnitude error
        :param z: the z band magnitude
        :param z_err: the z band magnitude error
        :param J: the J band magnitude
        :param J_err: the J band magnitude error
        :param H: the H band magnitude
        :param H_err: the H band magnitude error
        :param K: the K band magnitude
        :param K_err: the K band magnitude error
        :param grond_transmission: a file name specifying the GROND filter curves.
        If not provided, we will check the threeML data directory for it and then try to download it
        :param use_vega: to use vega for the H J K bands
        :param verbose: the verbose level of the plugin
        """

        if grond_transmission is None:

            # we will see if it is stored in the threeML data directory

            grond_filter_path = os.path.join(os.path.expanduser('~'), '.threeML', 'data')

            grond_filter_file = os.path.join(grond_filter_path, 'grond_filter_curves.txt')

            if file_existing_and_readable(grond_filter_file):

                grond = pd.read_table(grond_filter_file)

            else:

                # now we will try to download it

                self._download_filter_curves()

                if file_existing_and_readable(grond_filter_file):

                    grond = pd.read_table(grond_filter_file)


                else:

                    raise RuntimeError(
                        'You do not have the GROND transmission curves and they could not be obtained on line')



        else:

            # ok, instead you have provided a file path

            grond = pd.read_table(grond_transmission)

        # we will now build up the filters from inputs

        used_bands = collections.OrderedDict()

        if g is not None:
            assert g_err is not None, 'you must provide an error for the g band'
            used_bands['g'] = {'mag': g, 'err': g_err}

        if r is not None:
            assert r_err is not None, 'you must provide an error for the r band'
            used_bands['r'] = {'mag': r, 'err': r_err}

        if i is not None:
            assert i_err is not None, 'you must provide an error for the i band'
            used_bands['i'] = {'mag': i, 'err': i_err}

        if z is not None:
            assert z_err is not None, 'you must provide an error for the z band'
            used_bands['z'] = {'mag': z, 'err': z_err}

        if J is not None:
            assert J_err is not None, 'you must provide an error for the J band'
            used_bands['J'] = {'mag': J, 'err': J_err}

        if H is not None:
            assert H_err is not None, 'you must provide an error for the H band'
            used_bands['H'] = {'mag': H, 'err': H_err}

        if K is not None:
            assert K_err is not None, 'you must provide an error for the g band'
            used_bands['K'] = {'mag': K, 'err': K_err}

        assert len(used_bands) > 0, 'you have not loaded any data!'

        # read the GROND filter data



        columns = ['%sBand' % band for band in used_bands.keys()]

        transmission_curves = grond[columns].as_matrix().T  # type: np.ndarray

        wave_lengths = np.array([grond['A']] * transmission_curves.shape[0])  # [0]

        mag_system_to_use = []

        for itr, band in enumerate(_grond_filter_names):

            if band in used_bands:

                if use_vega:

                    # formally, GROND's HJK bands are vega

                    mag_system_to_use.append(_grond_magnitude_systems_vega[itr])

                else:

                    # sometimes the GCN data has been converted to AB, the user must know!

                    mag_system_to_use.append(_grond_magnitude_systems_ab[itr])

        # create the GROND filter set

        grond_filter = FilterSet(filter_names=used_bands.keys(),
                                 wave_lengths=wave_lengths,
                                 transmission_curves=transmission_curves,
                                 magnitude_systems=mag_system_to_use,
                                 wavesunits='nm')

        grond_data = PhotometryData(magnitudes=[band['mag'] for band in used_bands.itervalues()],
                                    magnitude_errors=[band['err'] for band in used_bands.itervalues()],
                                    filter_names=used_bands.keys())

        super(GRONDLike, self).__init__(name=name,
                                        photometry_data=grond_data,
                                        filter_set=grond_filter,
                                        verbose=verbose)

    def _download_filter_curves(self):

        print('\nThe GROND filter curves will now be downloaded if you have a working internet connection\n')

        grond_filter_path = os.path.join(os.path.expanduser('~'), '.threeML', 'data')

        grond_filter_file = os.path.join(grond_filter_path, 'grond_filter_curves.txt')

        # make the directory

        if_directory_not_existing_then_make(grond_filter_path)

        # go to the GROND website and download the filter curves

        response = urllib2.urlopen(threeML_config['grond']['GROND filter curve online location'])

        filter_data = response.readlines()

        # write the filter curves to a file

        with open(grond_filter_file, 'w') as f:
            for line in filter_data:
                f.write(line)

    @classmethod
    def from_file(cls, file):

        raise NotImplementedError('file reading has not been implemented')