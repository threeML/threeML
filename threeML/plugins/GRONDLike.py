import numpy as np
import pandas as pd
import collections

from threeML.plugins.PhotometryLike import PhotometryLike
from threeML.plugins.photometry.filter_set import FilterSet
from threeML.plugins.photometry.photometric_data import PhotometryData

_grond_filter_names = ['g', 'r', 'i', 'z', 'J', 'H', 'K']
_grond_magnitude_systems_vega = ['abmag', 'abmag','abmag', 'abmag', 'vega mag','vega mag', 'vega mag']
_grond_magnitude_systems_ab = ['abmag', 'abmag','abmag', 'abmag', 'abmag','abmag', 'abmag']

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




        assert grond_transmission is not None, 'do not be a fool'

        # read the GROND filter data

        grond = pd.read_table(grond_transmission)

        columns = ['%sBand'%band for band in used_bands.keys() ]

        transmission_curves = grond[columns].as_matrix().T #type: np.ndarray

        wave_lengths = np.array([grond['A']] * transmission_curves.shape[0])#[0]

        mag_system_to_use = []


        for itr, band in enumerate(_grond_filter_names):

            if band in used_bands:

                if use_vega:

                    mag_system_to_use.append(_grond_magnitude_systems_vega[itr])

                else:

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







        super(GRONDLike,self).__init__(name=name,
                                       photometry_data=grond_data,
                                       filter_set=grond_filter,
                                       verbose=verbose)





