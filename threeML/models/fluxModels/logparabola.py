from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import numpy
import math
import scipy.integrate
import operator

import collections


class LogParabola(SpectralModel):
    def setup(self):
            self.functionName        = "LogParabola"
            self.formula             = r'\begin{equation}f(E) = A E^{\gamma+\beta \log(E)}\end{equation}'
            self.parameters          = collections.OrderedDict()
            self.parameters['gamma'] = Parameter('gamma',-1.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
            self.parameters['beta'] = Parameter('beta',-0.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
            self.parameters['logA']     = Parameter('logA',-10,-40,20,1,fixed=False,nuisance=False,dataset=None,normalization=True)
            self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True)
    
            self.ncalls              = 0
    


            def integral(e1,e2):
                return self((e1+e2)/2.0)*(e2-e1)
            self.integral            = integral
    
    def __call__(self,e):
          
          self.ncalls             += 1
          piv                     = self.parameters['Epiv'].value
          gamma                = self.parameters['gamma'].value
          beta                    = self.parameters['beta'].value
          norm                     = pow(10, self.parameters['logA'].value)
          
          energies = numpy.array( e, ndmin=1, copy=False)          
 
          return norm * (energies/piv)**(gamma+beta*numpy.log10(energies/piv))
  
  

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


class LogParabolaEp(SpectralModel):
    def setup(self):
            self.functionName        = "LogParabolaEp"
            self.formula             = r'\begin{equation}f(E) = \frac{S_{p}}{E^{2}} (E/E_{p})^{-b \log{E/E_{p}}}\end{equation}'
            self.parameters          = collections.OrderedDict()
            self.parameters['b'] = Parameter('b',0.6,0,5,0.1,fixed=False,nuisance=False,dataset=None)
            self.parameters['Sp'] = Parameter('Sp',1,1e-5,1e5,0.1,normalization=True, fixed=False,nuisance=False,dataset=None)
            self.parameters['Ep']  = Parameter('Ep',300.0,1.0,1e6,100,fixed=False,nuisance=False,dataset=None)

            def integral(e1,e2):
                return self((e1+e2)/2.0)*(e2-e1)
            
            self.integral            = integral
    
    def __call__(self, e ):
          
          energies = numpy.array( e, ndmin=1, copy=False)
          
          b = self.parameters['b'].value
          Sp = self.parameters['Sp'].value
          Ep = self.parameters['Ep'].value
          
          eep = energies / Ep
          
          out = Sp / numpy.power( energies, 2 ) * numpy.power( eep, -b * numpy.log( eep ) )
          
          if(out.shape[0]==1):
          
            return out[0]
          
          else:
          
            return out  
