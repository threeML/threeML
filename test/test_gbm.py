import pytest
import numpy as np
import os

__author__ = 'drjfunk'

from threeML.plugins.FermiGBMTTELike import FermiGBMTTELike

from threeML.io.file_utils import within_directory

__example_dir = 'examples'


def test_gbm_tte_constructor():
    with within_directory(__example_dir):
        data_dir = os.path.join('gbm', 'bn080916009')

        src_selection = "0.-10."

        nai3 = FermiGBMTTELike('NAI3',
                               os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                               "-10-0, 100-150",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"), poly_order=-1)

        assert nai3.name == 'NAI3'

        assert nai3._active_interval == ('0.-10.',)
        assert nai3._startup == False
        assert nai3._verbose == True
