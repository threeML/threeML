from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import numpy
import math
import scipy.integrate
import operator
import numexpr

import collections


class LogParabola(SpectralModel):
    def __init__(self):
            self.functionName        = "LogParabola"
            self.formula             = r'\begin{equation}f(E) = A E^{\gamma+\beta \log(E)}\end{equation}'
            self.parameters          = collections.OrderedDict()
            self.parameters['gamma'] = Parameter('gamma',-1.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
            self.parameters['beta'] = Parameter('beta',-0.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
            self.parameters['A']     = Parameter('A',1.0,1e-10,1e10,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
            self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True)
    
            self.ncalls              = 0
    


            def integral(e1,e2):
                return self((e1+e2)/2.0)*(e2-e1)
            self.integral            = integral
    def __call__(self,energy):
          self.ncalls             += 1
          piv                     = self.parameters['Epiv'].value
          gamma                = self.parameters['gamma'].value
          beta                    = self.parameters['beta'].value

          return numpy.maximum(self.parameters['A'].value * numpy.power(energy/piv,gamma+(beta*numpy.log10(energy/piv))),1e-35)
  
  

    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
  #def energyFlux(self,e1,e2):
  #  a                        = self.parameters['gamma'].value
  #  piv                      = self.parameters['Epiv'].value
  #  if(a!=-2):
  #    def eF(e):
  #      return numpy.maximum(self.parameters['A'].value * numpy.power(e/piv,2-a)/(2-a),1e-30)
  #  else:
  #    def eF(e):
  #      return numpy.maximum(self.parameters['A'].value * numpy.log(e/piv),1e-30)
  #  pass
    
   # return (eF(e2)-eF(e1))*keVtoErg
pass
