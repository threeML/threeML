__author__ = 'drjfunk'

from threeML import *



from threeML.io.file_utils import within_directory


__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


#
# These tests simply check that the plugins with no instrumental software dependence, i.e.,
# those plugins which should be immediately available to the user
#
__example_dir = os.path.join(__this_dir__, '../../examples')




# download the data needed for the test

#gbm_data = download_GBM_trigger_data('bn080916009',detectors=['n3'],destination_directory=os.path.join(__example_dir,'gbm','bn080916009'),compress_tte=True)

def test_loading_ogip():

    with within_directory(__this_dir__):


        ogip = OGIPLike('test_ogip', observation='test.pha{1}')


def test_loading_xrt():

    with within_directory(__example_dir):

        xrt_dir = 'xrt'
        xrt = SwiftXRTLike("XRT", observation=os.path.join(xrt_dir, "xrt_src.pha"),
                           background=os.path.join(xrt_dir, "xrt_bkg.pha"),
                           response=os.path.join(xrt_dir, "xrt.rmf"),
                           arf_file=os.path.join(xrt_dir, "xrt.arf"))

# already tested
# def test_loading_gbm():
#
#
#
#     with within_directory(__example_dir):
#
#
#         data_dir = os.path.join('gbm', 'bn080916009')
#
#         src_selection = "0.-10."
#
#         # We start out with a bad background interval to demonstrate a few features
#
#         nai3 = FermiGBMTTELike('NAI3',
#                                os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
#                                "-10-0, 100-200",
#                                src_selection,
#                                rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v00.rsp2"), poly_order=2)
#
#
#
#
#
#     cleanup_downloaded_GBM_data(gbm_data)
#
#



