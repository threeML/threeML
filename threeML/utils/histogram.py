import numpy as np
import matplotlib.pyplot as plt

from threeML.utils.interval import IntervalSet, Interval
from threeML.io.step_plot import step_plot








class Histogram(IntervalSet):

    INTERVAL_TYPE = Interval

    def __init__(self,list_of_intervals,contents,errors=None,sys_errors=None, is_poisson=False):

        assert len(list_of_intervals) == len(contents), 'contents and intervals are not the same dimension '


        self._contents = contents

        if errors is not None:

            assert len(errors) == len(contents), 'contents and errors are not the same dimension '


            assert  is_poisson == False, 'cannot have errors and is_poisson True'


        self._errors = errors


        if sys_errors is not None:

            assert len(sys_errors) == len(contents),  'contents and errors are not the same dimension '


        self._sys_errors = sys_errors

        self._is_poisson = is_poisson





        super(Histogram, self).__init__(list_of_intervals)


    @property
    def errors(self):

        return self._errors

    @property
    def sys_errors(self):

        return self._sys_errors

    @property
    def contents(self):

        return self._contents


    @property
    def is_poisson(self):

        return self._is_poisson

    @classmethod
    def from_numpy_histogram(cls,hist,errors=None,sys_errors=None,is_poisson=False,**kwargs):
        """
        create a Histogram from a numpy histogram.
        Example:

            r = np.random.randn(1000)
            np_hist = np.histogram(r)
            hist = Histogram.from_numpy_histogram(np_hist)


        :param hist: a np.histogram tuple
        :param errors: list of errors for each bin in the numpy histogram
        :param sys_errors: list of systematic errors for each bin in the numpy histogram
        :param is_poisson: if the data is Poisson distributed or not
        :param kwargs: any kwargs to pass along
        :return: a Histogram object
        """

        # extract the contents and the bin edges

        contents = hist[0] #type: np.ndarray
        edges = hist[1] #type: np.ndarray

        # convert the edges to an interval set

        bounds = IntervalSet.from_list_of_edges(edges) #type: IntervalSet

        # return a histogram

        return cls(list_of_intervals=bounds,
                   contents=contents,
                   errors=errors,
                   sys_errors=sys_errors,
                   is_poisson=is_poisson,
                   **kwargs)


    def display(self,fill=False,fill_min=0.,x_label='x',y_label='y',**kwargs):

        fig, ax =  plt.subplots()

        step_plot(xbins=self.bin_stack,
                  y=self._contents,
                  ax=ax,
                  fill=fill,
                  fill_min=fill_min,
                  **kwargs)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig















