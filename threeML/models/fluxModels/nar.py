from threeML.models.tablemodel import FITSTableModel
import numpy
import collections
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator



class NAR(FITSTableModel):

    def setup(self):
        
        self.SetTableFile("/Users/jburgess/Research/3ML/threeml/models/fluxModels/MyTemplates/NAR.fits")

                
        self.functionName        = "Solar Flare Nuclear Excitation"
        self.formula             = r"Ask Nicola"
        self.parameters          = collections.OrderedDict()
        self.parameters['S'] = Parameter('S',2.6,2.59999,4,0.1,fixed=False,nuisance=False,dataset=None)
        
        self.integral = 1.
        self.ncalls              = 0
    
        
 
  
    def __call__(self,energy):
        self.ncalls             += 1
        S                       = self.parameters['S'].value
        
     
        return self._interpFunc((S,energy))
