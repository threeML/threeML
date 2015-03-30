from threeML.models.spatialmodel import SpatialModel #SpatialModel does not exist yet, needs to be written!!!
from threeML.models.Parameter import Parameter, SpatialParameter #SpatialParameter does not exist yet, needs to be implemented
import numpy as np

class Disk(SpatialModel):
    
    def setup(self):
        self.functionName        = "Disk"
        self.formula             = r'$f({\rm RA, Dec}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1}{\pi r^2} \, \left({\rm angsep} ({\rm RA, Dec, RA_0, Dec_0}) < r \right) $'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['radius'] = SpatialParameter('radius',0.1,0,20,0.01,fixed=False,nuisance=False,dataset=None)
        
        self.ncalls              = 0
    
    
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        radius                      = self.parameters['radius'](energy).value
        
        return np.power(180/np.pi,2)*1./(np.pi*radius**2) * (angsep(RA,Dec,RA0,Dec0)<radius)
