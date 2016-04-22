from threeML.plugins.GenericOGIPLike import GenericOGIPLike

__instrument_name = "Swift XRT"


class SwiftXRTLike(GenericOGIPLike):

    def __init__(self, name, phafile, bkgfile, rspfile, arffile):

        super(SwiftXRTLike, self).__init__(name, phafile, bkgfile, rspfile, arffile)
