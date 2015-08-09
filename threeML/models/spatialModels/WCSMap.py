#author L. Tibaldo (ltibaldo@slac.stanford.edu)

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter
import pyfits as pf
import numpy as np
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u

class WCSMap(SpatialModel):

    def setup(self,file):
        #assumes that file is a fits file with WCS map in hdu 0
        self.filename=file
        self.w=WCS(file)
        self.map=np.nan_to_num(pf.getdata(file,0))
        
        self.functionName        = "WCSMap"
        self.parameters          = collections.OrderedDict()
        self.parameters['Norm']  = Parameter('Norm',1.,0.,1.e5,0.1,fixed=True,nuisance=False,dataset=None)
        self.ncalls              = 0

    def __call__(self,RA,Dec):
        #you should make sure to oversample the original map when you call this function
        self.ncalls += 1
        Norm = self.parameters['Norm'].value
        #if the map is in Galactic coords need to convert from Celestial
        if w.wcs.lngtyp=='GLON'
            c = coord.ICRS(ra=RA, dec=Dec, unit=(u.degree, u.degree))
            lon = c.galactic.l
            lat = c.galactic.b
        else:
            lon, lat = RA, Dec
        px,py = self.w.wcs_world2pix(lon, lat, 1)
        #4-point linear interpolation to determine the values at the requested coords
        px_0=np.int(px)
        py_0=np.int(py)
        px_arr=np.array([px_0,px_0,px_0+1,px_0+1])
        py_arr=np.array([py_0,py_0+1,py_0,py_0+1])
        try:
            vals = self.map[py_arr,px_arr]
        except ValueError:
...         print "The WCS map in file {} is not defined at the sky coordinates requested by the user".format(self.filename)

        return Norm*vals
