from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
from threeML.exceptions import CustomExceptions

import numpy
import math
import scipy.integrate
import operator

import collections

class Band(SpectralModel):
    def setup(self):
        self.functionName       = "Band function [Band et al. 1993]"
        self.formula            = r'''
        \[f(E) = \left\{ \begin{eqnarray}
        K \left(\frac{E}{100 \mbox{ keV}}\right)^{\alpha} & \exp{\left(\frac{-E}{E_{c}}\right)} & \mbox{ if } E < (\alpha-\beta)E_{c} \\
        K \left[ (\alpha-\beta)\frac{E_{c}}{100}\right]^{\alpha-\beta}\left(\frac{E}{100 \mbox{ keV}}\right)^{\beta} & \exp{(\beta-\alpha)} & \mbox{ if } E \ge (\alpha-\beta)E_{c}
        \end{eqnarray}
        \right.
        \]
        '''

        self.parameters         = collections.OrderedDict()
        self.parameters['alpha'] = Parameter('alpha',-1.0,-5,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['beta']  = Parameter('beta',-2.0,-10,0,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['E0']    = Parameter('E0',500,10,1e5,10,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
    
        def integral(e1,e2):
            return self((e1+e2)/2.0)*(e2-e1)
        self.integral            = integral
    
  
  
    def __call__(self,e):
        #The input e can be either a scalar or an array
        #The following will generate a wrapper which will
        #allow to treat them in exactly the same way,
        #as they both were arrays, while keeping the output
        #in line with the input: if e is a scalar, the output
        #will be a scalar; if e is an array, the output will be an array
        energies                 = numpy.array(e,ndmin=1,copy=False)
        alpha                    = self.parameters['alpha'].value
        beta                     = self.parameters['beta'].value
        E0                       = self.parameters['E0'].value
        K                        = self.parameters['K'].value
        
        if(alpha < beta):
          raise CustomExceptions.ModelAssertionViolation("Alpha cannot be less than beta")
        
        out                      = numpy.zeros(energies.flatten().shape[0])
        idx                      = (energies < (alpha-beta)*E0)
        nidx                     = ~idx
        out[idx]                 = numpy.maximum(K*numpy.power(energies[idx]/100.0,alpha)*numpy.exp(-energies[idx]/E0),1e-30)
        out[nidx]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(energies[nidx]/100.0,beta),1e-30)
    
        #This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out                      = numpy.nan_to_num(out)
        if(out.shape[0]==1):
            return out[0]
        else:
            return out
 
