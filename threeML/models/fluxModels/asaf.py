from threeML.models.tablemodel import FITSTableModel
import numpy
import collections
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator



class AsafModel(FITSTableModel):

    def setup(self):
        
        self.SetTableFile("/Users/jburgess/Research/3ML/threeml/models/TableModel_grid256_res8_Nr2.fits")

                
        self.functionName        = "SubPhotosphericDissipation"
        self.formula             = r"Pe'er (2005)"
        self.parameters          = collections.OrderedDict()
        self.parameters['Tau'] = Parameter('Tau',1.,1.,10.,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['Gamma']     = Parameter('Gamma',50.,50.,500.,0.02,fixed=False,nuisance=False,dataset=None,normalization=False)
        self.parameters['LGRB']  = Parameter('LGRB',.1,100.,1e10,1,fixed=False)
        self.parameters['epsilon_d']  = Parameter('epsilon_d',.2,.2,.699999,1,fixed=False)
    
        self.ncalls              = 0
    
        
 
  
    def __call__(self,energy):
        self.ncalls             += 1
        tau                       = self.parameters['Tau'].value
        gamma                     = self.parameters['Gamma'].value
        LGRB                      = self.parameters['LGRB'].value
        ed                        = self.parameters['epsilon_d'].value
     
        return self._interpFunc((tau,gamma,LGRB,ed,energy))
