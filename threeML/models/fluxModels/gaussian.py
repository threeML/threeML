from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import scipy.special
import operator
import numpy


import collections


#test
class Gaussian(SpectralModel):

    def setup(self):
        self.functionName        = "Gaussian"
        self.formula             = r'$f(E) = A {\rm exp}\left(-\frac{(E-\mu)^2}{2\sigma^2} \right)$'
        self.parameters          = collections.OrderedDict()
        self.parameters['A']     = Parameter('A',1.,1.E-10,1.E10,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['mu'] = Parameter('mu',10.,1.,1E6,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['sigma'] = Parameter('sigma',1.,1E-2,1E1,0.1,fixed=False,nuisance=False,dataset=None)
            
        self.ncalls              = 0
        
        def integral(e1,e2):

            sigma                      = self.parameters['sigma'].value
            mu                         = self.parameters['mu'].value
            A                          = self.parameters['A'].value
            
    

            
            def f(x):

                term =  (mu - x)/(numpy.sqrt(2.)*sigma) 
                return -A* numpy.sqrt(numpy.pi/2.) * scipy.special.erf(term) * sigma
            return f(e2) - f(e1)
            
        self.integral            = integral
 
  
    def __call__(self,energy):
        self.ncalls             += 1
        sigma                      = self.parameters['sigma'].value
        mu                         = self.parameters['mu'].value
        A                          = self.parameters['A'].value
        return numpy.maximum( A * exp(-0.5 * (energy-mu)/(sigma) * (energy-mu)/(sigma)   ), 1e-30)
   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        sigma                      = self.parameters['sigma'].value
        mu                         = self.parameters['mu'].value
        A                          = self.parameters['A'].value

        
        def eF(x):
            term1 = (mu - x)/(numpy.sqrt(2.)*sigma) 
            term2 = -0.5 * (x-mu)/(sigma) * (x-mu)/(sigma)

            val = numpy.sqrt(numpy.pi/2.) * scipy.special.erf(term1) * sigma
            val += numpy.exp(term2) * sigma * sigma
             
            return -A*val

        return (eF(e2) - eF(e1))*self.keVtoErg

        

