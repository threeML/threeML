import numpy as np

from ogip import OGIPPluginPGstat
from threeML.plugin_prototype import PluginPrototype
from ogip import OGIPPHA


__instrument_name = "Fermi GBM (all detectors)"

# At the moment this is just another name for GenericOGIPLike
# In the future we might add functions to (for example) produce
# background spectra and so on

class FermiGBMLike(OGIPPluginPGstat, PluginPrototype):

    def __init__(self, name, phafile, bkgfile, rspfile, arffile=None):
        """
        If the input files are PHA2 files, remember to specify the spectrum number, for example:
        FermiGBMLike("GBM","spectrum.pha{2}","bkgfile.bkg{2}","rspfile.rsp{2}")
        to load the second spectrum, second background spectrum and second response.
        """

        OGIPPluginPGstat.__init__(self, name, rspfile, bkgfile=bkgfile, phafile=phafile)

        self.phafile = OGIPPHA(phafile, filetype='observed')
        self.exposure = self.phafile.getExposure()

        self.bkgfile = OGIPPHA(bkgfile, filetype="background")

        # Start with an empty mask (the user will overwrite it using the
        # setActiveMeasurement method)
        mask = np.asarray(np.ones(self.phafile.getRates().shape), np.bool)

        # Get the counts for this spectrum
        counts = (self.phafile.getRates() * self.exposure)

        # Get the background counts for this spectrum
        bkgCounts = (self.bkgfile.getRates() * self.exposure)

        # Get the error on the background counts
        bkgErr = self.bkgfile.getRatesErrors() * self.exposure

        self._initialSetup(mask, counts, bkgCounts, self.exposure, bkgErr)