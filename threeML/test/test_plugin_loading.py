__author__ = 'drjfunk'

from threeML import *



from threeML.io.file_utils import within_directory


__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))

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





