from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator
import numpy


import collections



class ExponentialCutoff(SpectralModel):

    def setup(self):
        self.functionName        = "ExponentialCutoff"
        self.formula             = r'$f(E) = A {\rm exp}\left(-E/E_{\rm fold}   \right)$'
        self.parameters          = collections.OrderedDict()
        self.parameters['A']     = Parameter('A',1.,1.E-10,1.E10,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['logEfold'] = Parameter('logEfold',2,0,20,0.1,fixed=False,nuisance=False,dataset=None)
            
        self.ncalls              = 0
    
        def integral(e1,e2):
            eFold                      = pow(10,self.parameters['logEfold'].value)
            A                          = self.parameters['A'].value

            
            def f(x):

                return -A * eFold * numpy.exp(-x/eFold)

            return f(e2) - f(e1)
            
        self.integral            = integral
 
  
    def __call__(self,e):
        self.ncalls             += 1
        eFold                      = pow(10, self.parameters['logEfold'].value)
        A                          = self.parameters['A'].value
        
        energies = numpy.array( e, ndmin=1, copy=False)
        
        return numpy.maximum( numpy.exp(-energies/eFold), 1e-30)
   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        eFold                      = pow(10,self.parameters['logEfold'].value)
        A                          = self.parameters['A'].value

        def eF(x):

            return -numpy.exp(-x/eFold)*(x*eFold + (eFold*eFold ))

        return (eF(e2) - eF(e1))*self.keVtoErg

        


