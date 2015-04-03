#Author: L. Tibaldo, ltibaldo@slac.stanford.edu

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter

import numpy as np
import collections

class Ellipse(SpatialModel):
    
    def setup(self):
        self.functionName        = "Ellipse"
        self.formula             = r'$f({\rm RA, Dec}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1}{\pi a^2 \sqrt{1-e^2}} \, \left[ \left(\frac{\cos\theta \,{\rm RA} + \sin\theta\, {\rm Dec} - {\rm RA_0}}{a^2} + \frac{-\sin\theta\, {\rm RA} + \cos \theta \,{\rm Dec}- {\rm Dec_0}}{a^2 (1-e^2)}\right) < 1 \right]$'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['maj_saxis'] = SpatialParameter('maj_saxis',0.1,0,10,0.01,fixed=False,nuisance=False,dataset=None)
        self.parameters['eccentricity'] = SpatialParameter('eccentricity',0.7,0,1,0.01,fixed=False,nuisance=False,dataset=None)
        self.parameters['angle'] = SpatialParameter('angle',0.,0,180,1.,fixed=False,nuisance=False,dataset=None)
        
        self.ncalls              = 0
    
    
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        maj_saxis                   = self.parameters['maj_saxis'].getValue(energy)
        eccentricity                = self.parameters['eccentricity'].getValue(energy)
        angle                       = self.parameters['angle'].getValue(energy)
        
        s=np.sin(np.deg2rad(angle))
        c=np.cos(np.deg2rad(angle))
        return np.power(180/np.pi,2)*1./(np.pi*maj_saxis**2*np.sqrt(1-eccentricity**2))*(np.power(c*RA+s*Dec-c*RA0-s*Dec0,2)/maj_saxis**2+np.power(-s*RA+c*Dec+s*RA0-c*Dec0,2)/(maj_saxis**2*(1-eccentricity**2))<1)


    def integratedFlux(self,energy):
    
        return 1.