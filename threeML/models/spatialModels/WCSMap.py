#author L. Tibaldo (ltibaldo@slac.stanford.edu)

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter
import pyfits as pf
import numpy as np
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u

def return_lonlat(RA,Dec,w):
        #if the map is in Galactic coords need to convert from Celestial
        if w.wcs.lngtyp=='GLON':
            c = coord.ICRS(ra=RA, dec=Dec, unit=(u.degree, u.degree))
            lon = c.galactic.l
            lat = c.galactic.b
        else:
            lon, lat = RA, Dec

        return lon, lat


def fp_linear_interp(pos,map,filename):
        #4-point linear interpolation to determine the values at the requested coords
        px_0=pos[-1].astype('int')
        py_0=pos[-2].astype('int')
        px_arr=np.array([px_0,px_0,px_0+1,px_0+1])
        py_arr=np.array([py_0,py_0+1,py_0,py_0+1])
        b=pos[-1]-px_0
        a=1.-b
        d=pos[-2]-py_0
        c=1.-d
        if len(pos)==2:
            fp_pos=[py_arr,px_arr]
        else:
            fp_pos=[4*[pos[-3],py_arr,px_arr]]
        try:
            vals = map[fp_pos]
        except ValueError:
            print "The WCS map in file {} is not defined at the sky coordinates requested by the user".format(filename)
        vals=vals*np.array([a*c,a*d,b*c,b*d])
        vals=np.sum(vals,axis=0)

        return vals


class WCSSpatialMap(SpatialModel):

    def setup(self,file):
        #assumes that file is a fits file with WCS map in hdu 0
        self.filename=file
        self.w=WCS(file)
        self.map=np.nan_to_num(pf.getdata(file,0))
        
        self.functionName        = "WCSSpatialMap"
        self.parameters          = collections.OrderedDict()
        self.parameters['Norm']  = Parameter('Norm',1.,0.,1.e5,0.1,fixed=True,nuisance=False,dataset=None)
        self.ncalls              = 0

    def __call__(self,RA,Dec):
        #you should make sure to oversample the original map when you call this function
        self.ncalls += 1
        Norm = self.parameters['Norm'].value
        #if the map is in Galactic coords need to convert from Celestial
        lon, lat = return_lonlat(RA,Dec,self.w)
        px,py = self.w.wcs_world2pix(lon, lat, 1)
        pos=[py,px]
        vals=fp_linear_interp(vec,self.map,self.filename)

        return Norm*vals


class WCSMapCube(SpatialModel):

    def setup(self,file):
        #assumes that file is a fits file with WCS map in hdu 0
        self.filename=file
        self.w=WCS(file)
        self.cube=np.nan_to_num(pf.getdata(file,0))
        self.energies=pf.getdata(file,1)['Energy']
        
        self.functionName        = "WCSMapCube"
        self.parameters          = collections.OrderedDict()
        self.parameters['Norm']  = Parameter('Norm',1.,0.,1.e5,0.1,fixed=True,nuisance=False,dataset=None)
        self.ncalls              = 0

    def __call__(self,RA,Dec,energy):
        #you should make sure to oversample the original map when you call this function
        self.ncalls += 1
        Norm = self.parameters['Norm'].value
        #if the map is in Galactic coords need to convert from Celestial
        lon, lat = return_lonlat(RA,Dec,self.w)
        px,py = self.w.wcs_world2pix(lon, lat, 0, 1)
        #determine values at requested energies by PL interp on the nearest available in the model
        E_0=np.max(np.where(energies<energy[:,np.newaxis]),axis=1)
        pos_0=[E_0,py,px]
        vals_0=fp_linear_interp(pos_0,self.cube,self.filename)
        pos_1=[E_0+1,py,px]
        vals_1=fp_linear_interp(pos_1,self.cube,self.filename)
        logE0=np.log(self.energies[E_0:np.newaxis,np.newaxis])
        logE1=np.log(self.energies[E_0+1:np.newaxis,np.newaxis])
        gamma=-(np.log(vals_1)-np.log(vals_0))/(logE1-logE0)
        vals=vals_0-gamma*(np.log(energy)-logE0)
        vals=np.exp(vals)

        return Norm*vals
