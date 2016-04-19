#author H. Zhou (hzhou1@mtu.edu)

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter
import numpy as np
import scipy.stats
from angsep import angsep

import collections


class Diffusion(SpatialModel):
    def setup(self):
        self.functionName        = "Diffusion"
        self.formula             = r'$f({\rm RA, Dec}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1.2154}{\sqrt{\pi^3} \rdiff ({\rm angsep} ({\rm RA, Dec, RA_0, Dec_0})+0.06 \rdiff)} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm RA, Dec, RA_0, Dec_0})}{\rdiff^2} \right) $'
        self.parameters          = collections.OrderedDict()
        self.parameters['RA0']     = Parameter('RA0',1.,0,360,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Dec0']     = Parameter('Dec0',1.,-90,90,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['rdiff'] = SpatialParameter('rdiff',1.,0,20,0.01,fixed=False,nuisance=False,dataset=None)
        self.parameters['delta'] = SpatialParameter('delta',0.5,0.3,0.6,0.01,fixed=True,nuisance=False,dataset=None)
            
        self.ncalls              = 0

  
    def __call__(self,RA,Dec,energy):
        self.ncalls             += 1
        RA0                         = self.parameters['RA0'].value
        Dec0                        = self.parameters['Dec0'].value
        #energy in kev -> TeV
        rdiff                       = self.parameters['rdiff'].value * np.power(np.divide(energy,2e10),(self.parameters['delta'].value - 1.)/2.*(0.54+0.046*np.log10(np.divide(energy,1e9))))
        
        return np.maximum( np.power(180/np.pi,2)*1.2154/(np.pi * np.sqrt(np.pi) * rdiff * (angsep(RA,Dec,RA0,Dec0) + 0.06 * rdiff)) * np.exp(-1. * np.power(angsep(RA,Dec,RA0,Dec0),2)/rdiff**2), 1e-30)

    def integratedFlux(self,energy):
    
        return 1.
    
    def getBoundaries(self):
        
        #Truncate the function at the max of rdiff allowed
        
        maxRdiff = self.parameters['rdiff'].maxValue
        
        minDec = max(-90.,self.parameters['Dec0'].value - maxRdiff)
        maxDec = min(90.,self.parameters['Dec0'].value + maxRdiff)

        maxAbsDec = max(np.absolute(minDec),np.absolute(maxDec))
        if maxAbsDec > 89. or maxRdiff/np.cos(maxAbsDec*np.pi/180.) >= 180.:
            minRa = 0.
            maxRa = 360.
        else:
            minRa = self.parameters['RA0'].value - maxRdiff/np.cos(maxAbsDec*np.pi/180.)
            maxRa = self.parameters['RA0'].value + maxRdiff/np.cos(maxAbsDec*np.pi/180.)
            if minRa < 0.:
                minRa = minRa + 360.
            elif maxRa > 360.:
                maxRa = maxRa - 360.
        
        return minRa, maxRa, minDec, maxDec
                      



        

