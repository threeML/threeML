import numpy as np
import pandas as pd

from threeML.plugins.PhotometryLike import PhotometryLike
from threeML.plugins.photometry.filter_set import FilterSet
from threeML.plugins.photometry.photometric_data import PhotometryData

_grond_filters = ['gBand', 'rBand', 'iBand', 'zBand', 'JBand', 'HBand', 'KBand']
_grond_filter_names = ['g', 'r', 'i', 'z', 'J', 'H', 'K']
_grond_magnitude_systems = ['abmag', 'abmag','abmag', 'abmag', 'vega mag','vega mag', 'vega mag']

class GRONDLike(PhotometryLike):

    def __init__(self, name, observation, grond_transmission, verbose=True):




        # read the GROND filter data

        grond = pd.read_table(grond_transmission)

        transmission_curves = grond[_grond_filters].as_matrix()

        wave_lengths = np.array([grond['A']] * len(_grond_filters))[0]

        # create the GROND filter set

        grond_filter = FilterSet(filter_names=_grond_filter_names,
                                 wave_lengths=wave_lengths,
                                 transmission_curves=transmission_curves,
                                 magnitude_systems=_grond_magnitude_systems,
                                 wavesunits='nm')

        grond_data = PhotometryData(magnitudes=None,
                                    magnitude_errors=None,
                                    filter_names=_grond_filter_names)







        super(GRONDLike,self).__init__(name=name,
                                       photometry_data=grond_data,
                                       filter_set=grond_filter,
                                       verbose=verbose)





