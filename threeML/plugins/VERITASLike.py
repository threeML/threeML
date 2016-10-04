__author__ = 'giacomov'

from threeML.plugins.OGIPLike import OGIPLike

__instrument_name = "VERITAS"

class VERITASLike(OGIPLike):

    def __init__(self, name, udara_style_root_file):

        super(VERITASLike, self).__init__(name, {})