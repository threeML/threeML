import warnings

import numpy as np

from threeML.plugin_prototype import PluginPrototype
from threeML.plugins.ogip import OGIPPHA, OGIPPluginCash

__instrument_name = "All OGIP-compliant instruments"


class GenericOGIPLike(OGIPPluginCash, PluginPrototype):

    def __init__(self, name, phafile, bkgfile, rspfile, arffile=None):
        """
        If the input files are PHA2 files, remember to specify the spectrum number, for example:
        FermiGBMLike("GBM","spectrum.pha{2}","bkgfile.bkg{2}","rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        """

        OGIPPluginCash.__init__(self, name, rspfile, arffile=arffile, bkgfile=bkgfile, phafile=phafile)

        self.phafile = OGIPPHA(phafile, filetype='observed')
        exposure = self.phafile.getExposure()
        self.bkgfile = OGIPPHA(bkgfile, filetype="background")

        # Start with an empty mask (the user will overwrite it using the
        # setActiveMeasurement method)
        mask = np.asarray(np.ones(self.phafile.getRates().shape), np.bool)

        # Get the counts for this spectrum
        counts = (self.phafile.getRates() * exposure)

        # Get the background counts for this spectrum
        bkgCounts = (self.bkgfile.getRates() * exposure)

        self._initialSetup(mask, counts, bkgCounts, exposure)