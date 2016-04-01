from threeML.models.tablemodel import NumpyTableModel
from threeML.models.Parameter import Parameter
import numpy
import math
import scipy.integrate
import operator

import collections

class BandTable(NumpyTableModel):
    def setup(self):

        self.SetTableFile("/Users/jburgess/Research/3ML/threeml/models/fluxModels/band.npz")


        
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
        self.parameters['alpha'] = Parameter('alpha',-1.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['beta']  = Parameter('beta',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['E0']    = Parameter('E0',500,10,1e5,50,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
    
    
  
  
    def __call__(self,e):
        #The input e can be either a scalar or an array
        #The following will generate a wrapper which will
        #allow to treat them in exactly the same way,
        #as they both were arrays, while keeping the output
        #in line with the input: if e is a scalar, the output
        #will be a scalar; if e is an array, the output will be an array
    
        alpha                    = self.parameters['alpha'].value
        beta                     = self.parameters['beta'].value
        E0                       = self.parameters['E0'].value
        K                        = self.parameters['K'].value
        
        return self._interpFunc((K,E0,alpha,beta,e))
    
        
 
