from __future__ import division
from past.utils import old_div
import pytest
from threeML.utils.interval import IntervalSet
from threeML.utils.histogram import Histogram
from threeML import *
from threeML.io.file_utils import within_directory
import numpy as np
import os


__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def is_within_tolerance(truth, value, relative_tolerance=0.01):
    assert truth != 0

    if abs(old_div((truth - value), truth)) <= relative_tolerance:

        return True

    else:

        return False


def test_hist_constructor():

    with within_directory(__this_dir__):

        bins = [-3, -2, -1, 0, 1, 2, 3]

        bounds = IntervalSet.from_list_of_edges(bins)

        contents = np.ones(len(bins) - 1)

        hh1 = Histogram(bounds, contents, is_poisson=True)

        assert hh1.is_poisson == True

        assert len(hh1) == len(bins) - 1

        hh1.display()

        # rnum = np.loadtxt('test_hist_data.txt')
        #
        #
        # #rnum = np.random.randn(1000)
        # hrnum = np.histogram(rnum, bins=bins, normed=False)
        # hh2 = Histogram.from_numpy_histogram(hrnum, is_poisson=True)

        # hh3 = Histogram.from_entries(bounds,rnum)
        #
        # assert hh2==hh3

        hh4 = Histogram(bounds, contents, errors=contents)

        assert hh4.is_poisson == False

        with pytest.raises(AssertionError):

            hh4 = Histogram(bounds, contents, errors=contents, is_poisson=True)


def test_hist_addition():

    bins = [-3, -2, -1, 0, 1, 2, 3]

    bounds = IntervalSet.from_list_of_edges(bins)

    contents = np.ones(len(bins) - 1)

    hh1 = Histogram(bounds, contents, is_poisson=True)

    hh2 = hh1 + hh1

    assert np.all(hh2.contents == 2 * hh1.contents)

    hh3 = Histogram(bounds, contents, errors=contents)

    hh4 = hh3 + hh3

    with pytest.raises(AssertionError):

        hh3 + hh1

    hh5 = Histogram(bounds, contents, errors=contents, sys_errors=contents)

    hh6 = hh5 + hh5

    hh7 = hh5 + hh3

    hh8 = hh3 + hh5
