__author__ = 'drjfunk'

from threeML import *



from threeML.io.file_utils import within_directory


__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


#
# These tests simply check that the plugins with no instrumental software dependnece, i.e.,
# those plugins which should be immediately available to the user
#

def test_loading_ogip():

    with within_directory(__this_dir__):


        ogip = OGIPLike('test_ogip', pha_file='test.pha{1}')


def test_loading_xrt():

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/'))




    with within_directory(datadir):

        xrt_dir = 'xrt'
        xrt = SwiftXRTLike("XRT", pha_file=os.path.join(xrt_dir, "xrt_src.pha"),
                           bak_file=os.path.join(xrt_dir, "xrt_bkg.pha"),
                           rsp_file=os.path.join(xrt_dir, "xrt.rmf"),
                           arf_file=os.path.join(xrt_dir, "xrt.arf"))


def test_loading_gbm():
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/'))

    with within_directory(datadir):
        data_dir = os.path.join('gbm', 'bn080916009')

        src_selection = "0.-10."

        # We start out with a bad background interval to demonstrate a few features

        nai3 = FermiGBMTTELike('NAI3',
                               os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                               "-10-0, 100-200",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"), poly_order=2)











