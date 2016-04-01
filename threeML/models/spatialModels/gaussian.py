#author L. Tibaldo (ltibaldo@slac.stanford.edu)

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter
import numpy as np
import scipy.stats
from angsep import angsep

import collections


class SimpleGaussian(SpatialModel):

    def setup(self):
        self.functionName        = "Gaussian"
        self.formula             = r'$f({\rm RA, Dec}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1}{2\pi \sigma^2} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm RA, Dec, RA_0, Dec_0})}{2\sigma^2} \right) $'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['sigma'] = SpatialParameter('sigma',0.1,0,20,0.01,fixed=False,nuisance=False,dataset=None)
            
        self.ncalls              = 0

  
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        sigma                       = self.parameters['sigma'].getValue(energy)
        
        return np.maximum( np.power(180/np.pi,2)*1./(2 * np.pi * sigma**2) * np.exp(-0.5 * np.power(angsep(RA,Dec,RA0,Dec0),2)/sigma**2), 1e-30)

    def integratedFlux(self,energy):
    
        return 1.
    
    def getBoundaries(self):
        
        #Truncate the gaussian at 2 times the max of sigma allowed
        
        maxSigma = self.parameters['sigma'].maxValue
        
        minDec = max(-90.,self.parameters['Dec0'].value - 2 * maxSigma)
        maxDec = min(90.,self.parameters['Dec0'].value + 2 * maxSigma)

        maxAbsDec = max(np.absolute(minDec),np.absolute(maxDec))
        if maxAbsDec > 89. or 2*maxSigma/np.cos(maxAbsDec*np.pi/180.) >= 180.:
            minRa = 0.
            maxRa = 360.
        else:
            minRa = self.parameters['RA0'].value - 2*maxSigma/np.cos(maxAbsDec*np.pi/180.)
            maxRa = self.parameters['RA0'].value + 2*maxSigma/np.cos(maxAbsDec*np.pi/180.)
            if minRa < 0.:
                minRa = minRa + 360.
            elif maxRa > 360.:
                maxRa = maxRa - 360.
        
        return minRa, maxRa, minDec, maxDec
                      


class MultiVariateGaussian(SpatialModel):
    
    def setup(self):
        self.functionName        = "Multivariate Gaussian"
        self.formula             = r'$f(\vec{x}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1}{2\pi \sqrt{\det{\Sigma}}} \, {\rm exp}\left( -\frac{1}{2} (\vec{x}-\vec{x}_0)^\intercal \cdot \Sigma^{-1}\cdot (\vec{x}-\vec{x}_0)\right) \\ \vec{x}_0 = ({\rm RA}_0,{\rm Dec}_0)\\ \Lambda = \left( \begin{array}{cc} \sigma^2 & 0 \\ 0 & \sigma^2 (1-e^2) \end{array}\right) \\ U = \left( \begin{array}{cc} \cos \theta & -\sin \theta \\ \sin \theta & cos \theta \end{array}\right) \\\Sigma = U\Lambda U^\intercal$'
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
        sigma                       = self.parameters['sigma'].getValue(energy)
        eccentricity                = self.parameters['eccentricity'].getValue(energy)
        angle                       = np.deg2rad(self.parameters['angle'].getValue(energy))
        
        
        sigma1=sigma**2
        sigma2=sigma1*(1-eccentricity**2)
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        cov=np.dot(rot,np.dot(np.array([[sigma1,0],[0,sigma2]]),rot.T))
        return np.maximum(np.power(180/np.pi,2)*scipy.stats.multivariate_normal.pdf(np.array([Dec,RA]).T, mean=np.array([RA0,Dec0]), cov=cov), 1e-30)
   
    def integratedFlux(self,energy):
                                                 
        return 1.



        

