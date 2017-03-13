import numpy as np
import pandas as pd

from threeML.io.rich_display import display
from threeML.utils.time_interval import TimeInterval, TimeIntervalSet

class PhotometryData(object):
    def __init__(self, magnitudes, magnitude_errors, filter_names, tstart=None, tstop=None):
        """
        A container to standardize the storing of photometric data
        on 3ML.

        :param magnitudes: observed magnitudes
        :param magnitude_errors: observed magnitude errors
        :param filter_names: filter names associated with the magnitudes
        """
        self._magnitudes = np.array(magnitudes)

        self._magnitude_errors = np.array(magnitude_errors)

        self._filter_names = np.array(filter_names)

        self._time = None

        if tstart is not None:

            assert tstop is not None, 'if you specify a start time, you must specify a stop time'


            self._time = TimeInterval(start=tstart, stop=tstop)


    @property
    def time(self):

        return self._time

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




class PhotometryDataSet(object):

    def __init__(self, list_of_photometric_data, reference_time=0.):
        # type: (list(PhotometryData), float) -> None

        self._list_of_photometric_data = list_of_photometric_data

        self._reference_time = reference_time


        # see if any intervals have time

        time_exists =[data.time is not None for data in self._list_of_photometric_data]

        if np.any(time_exists):

            # demand they all have time
            assert np.all(time_exists)

            self._time = TimeInterval([data.time for data in self._list_of_photometric_data])





    def __getitem__(self, item):

        return self._list_of_photometric_data[item]

    def __len__(self):

        return len(self._list_of_photometric_data)

    @property
    def magnitudes(self):

        return np.array([data.magnitudes for data in self._list_of_photometric_data])

    @property
    def magnitude_errors(self):
        return np.array([data.magnitude_errors for data in self._list_of_photometric_data])


    @property
    def filter_names(self):

        return self._list_of_photometric_data[0].filter_names


    @property
    def n_bands(self):
        return self._list_of_photometric_data[0].n_band



    @property
    def time(self):

        return self._time