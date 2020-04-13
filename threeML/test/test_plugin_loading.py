__author__ = "drjfunk"

from threeML.plugins.OGIPLike import OGIPLike
from threeML.plugins.SwiftXRTLike import SwiftXRTLike
import os
from .conftest import get_test_datasets_directory
from threeML.io.file_utils import within_directory

#
# These tests simply check that the plugins with no instrumental software dependence, i.e.,
# those plugins which should be immediately available to the user
#
datasets_dir = get_test_datasets_directory()


def test_loading_ogip():

    with within_directory(datasets_dir):

        _ = OGIPLike("test_ogip", observation="test.pha{1}")


def test_loading_xrt():

    with within_directory(datasets_dir):

        xrt_dir = "xrt"
        xrt = SwiftXRTLike(
            "XRT",
            observation=os.path.join(xrt_dir, "xrt_src.pha"),
            background=os.path.join(xrt_dir, "xrt_bkg.pha"),
            response=os.path.join(xrt_dir, "xrt.rmf"),
            arf_file=os.path.join(xrt_dir, "xrt.arf"),
        )
