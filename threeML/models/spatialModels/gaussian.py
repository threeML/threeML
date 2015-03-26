#from threeML.models.spectralmodel import SpectralModel
from threeML.models.spatialmodel import SpatialModel #SpatialModel does not exist yet, needs to be written!!!
from threeML.models.Parameter import Parameter, SpatialParameter #SpatialParameter does not exist yet, needs to be implemented
import numpy as np
import scipy.stats
import math

import collections


class Gaussian(SpatialModel):

    def setup(self):
        self.functionName        = "Gaussian"
        self.formula             = r'$f(\vec{x}) = \frac{1}{2\pi \sigma^2} \, {\rm exp}\left(-\frac{(\vec{x}-\vec{x}_0)^2}{2\sigma^2} \right) \\ \vec{x}_0 = ({\rm RA}_0,{\rm Dec}_0)$'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['sigma'] = SpatialParameter('sigma',0.1,0,20,0.01,fixed=False,nuisance=False,dataset=None)
            
        self.ncalls              = 0

  
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        sigma                       = self.parameters['sigma'](energy).value
        
        return numpy.maximum( numexpr.evaluate("1./(2 * pi * sigma**2) * exp(-0.5 * ((RA-RA0)**2 + (Dec - Dec0)**2)/sigma**2)"), 1e-30)


class MultiVariateGaussian(SpatialModel):
    
    def setup(self):
        self.functionName        = "Multivariate Gaussian"
        self.formula             = r'$f(\vec{x}) = \frac{1}{2\pi \sqrt{\det{\Sigma}}} \, {\rm exp}\left( -\frac{1}{2} (\vec{x}-\vec{x}_0)^\intercal \cdot \Sigma^{-1}\cdot (\vec{x}-\vec{x}_0)\right) \\ \vec{x}_0 = ({\rm RA}_0,{\rm Dec}_0)\\ \Lambda = \left( \begin{array}{cc} \sigma & 0 \\ 0 & \sigma \sqrt{1-e^2} \end{array}\right) \\ U = \left( \begin{array}{cc} \cos \theta & -\sin \theta \\ \sin \theta & cos \theta \end{array}\right) \\\Sigma = U\Lambda U^\intercal$'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['sigma'] = SpatialParameter('sigma',0.1,0,10,0.01,fixed=False,nuisance=False,dataset=None)
        self.parameters['eccentricity'] = SpatialParameter('eccentricity',0.7,0,1,0.01,fixed=False,nuisance=False,dataset=None)
        self.parameters['angle'] = SpatialParameter('angle',0.,0,180,1.,fixed=False,nuisance=False,dataset=None)
        
        
        self.ncalls              = 0
    
    
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        sigma                       = self.parameters['sigma'](energy).value
        eccentricity                = self.parameters['eccentricity'](energy).value
        angle                       = np.deg2rad(self.parameters['angle'](energy).value)
        
        
        sigma_1=sigma
        sigma_2=sigma*math.sqrt(1-eccentricity**2)
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        cov=np.dot(rot,np.dot(np.array([[sigma_1,0],[0,sigma_2]]),rot.T))
        return numpy.maximum(scipy.stats.multivariate_normal.pdf(np.array([RA,Dec]).T, mean=np.array([RA0,Dec0]).T, cov=cov), 1e-30)
   
  



        

