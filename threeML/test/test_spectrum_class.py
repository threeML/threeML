import pytest
from threeML.plugins.spectrum.binned_spectrum import BinnedSpectrum
from threeML.plugins.spectrum.binned_spectrum_set import BinnedSpectrumSet


from threeML.io.file_utils import within_directory

import os

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def test_spectrum_constructor():
    with within_directory(__this_dir__):

        pass
