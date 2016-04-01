from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator

import collections



class Blackbody(SpectralModel):

    def setup(self):
        self.functionName        = "Blackbody"
        self.formula             = r'$f(E) =  A E^{2}\frac{1}{{\rm exp} (E/kT) -1}$'
        self.parameters          = collections.OrderedDict()
        self.parameters['A'] = Parameter('A',100.,1.,1E6,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['kT'] = Parameter('kT',30.,1.,1E6,0.1,fixed=False,nuisance=False,dataset=None)
        
        
            
        Self.ncalls              = 0
    
#        def integral(e1,e2):
#            # No analytic expression
#            pass
#            
#        self.integral            = None
 
  
    def __call__(self,energy):
        self.ncalls             += 1
        A                        = self.parameters['A'].value
        kT                       = self.parameters['kT'].value
        
        return numpy.maximum( A*energyx**2.*power( exp(energy/kT) -1., -1.), 1e-30)
   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        pass


        #return (eF(e2)-eF(e1))*keVtoErg

