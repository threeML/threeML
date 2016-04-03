from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator
#import numexpr
import numpy


import collections



class PowerLaw(SpectralModel):

    def setup(self):
        self.functionName        = "Powerlaw"
        self.formula             = r'\begin{equation}f(E) = A (E / E_{piv})^{\gamma}\end{equation}'
        self.parameters          = collections.OrderedDict()
        self.parameters['gamma'] = Parameter('gamma',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['logA']     = Parameter('logA',-4,-40,30,0.1,fixed=False,nuisance=False,dataset=None,normalization=False)
        self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True,unit='keV')
    
        self.ncalls              = 0
    
        def integral(e1,e2):
            a                      = self.parameters['gamma'].value
            piv                    = self.parameters['Epiv'].value
            norm                   = pow(10, self.parameters['logA'].value)
      
            if(a!=-1):
                def f(energy):
                    return norm * energy * math.pow(energy/piv,a)/(a+1)
            else:
                def f(energy):
                    return norm * piv * math.log(energy)
                    
            return f(e2)-f(e1)
        self.integral            = integral
 
  
    def __call__(self,e):
        self.ncalls             += 1
        piv                      = self.parameters['Epiv'].value
        norm                     = pow(10, self.parameters['logA'].value)
        gamma                    = self.parameters['gamma'].value
        
        energies = numpy.array( e, ndmin=1, copy=False)

        return numpy.maximum( norm * (energies/piv)**gamma, 1e-100)
   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        a                        = self.parameters['gamma'].value
        piv                      = self.parameters['Epiv'].value
        norm                     = pow(10, self.parameters['logA'].value)
        if(a!=-2):
            def eF(e):
                return numpy.maximum( norm * numpy.power(e/piv,2-a)/(2-a), 1e-100 )
        else:
            def eF(e):
                return numpy.maximum( norm * numpy.log(e/piv), 1e-100 )
   
    
        return (eF(e2)-eF(e1))*self.keVtoErg

