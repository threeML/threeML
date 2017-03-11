import numpy as np
import pandas as pd

from threeML.io.rich_display import display

class PhotometryData(object):
    def __init__(self, magnitudes, magnitude_errors, filter_names):
        """
        A container to standardize the storing of photometric data
        on 3ML.

        :param magnitudes: observed magnitudes
        :param magnitude_errors: observed magnitude errors
        :param filter_names: filter names associated with the magnitudes
        """
        self._magnitudes = np.array(magnitudes)

        self._errors = np.array(magnitude_errors)

        self._filter_names = np.array(filter_names)

    @property
    def filter_names(self):
        """

        :return: the filter names
        """
        return self._filter_names

    @property
    def magnitudes(self):
        """

        :return: the magnitudes
        """
        return self._magnitudes

    @property
    def magnitude_errors(self):
        """

        :return: the magnitude errors
        """
        return self._magnitude_errors

    @property
    def n_bands(self):
        """

        :return: the number of bands/filters
        """
        return self._filter_names.shape[0]

    def display(self):
        """
        display the data
        :return:
        """
        df = pd.DataFrame(index=self._filter_names,
                          data={'magnitudes': self._magnitudes,
                                'magnitude  errors': self._magnitude_errors})

        display(df)

    @classmethod
    def from_fits(cls):

        raise NotImplementedError('Not ready yet')
