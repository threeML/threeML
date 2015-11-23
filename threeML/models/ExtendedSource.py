#Author: L. Tibaldo, ltibaldo@slac.stanford.edu

from SourceModel import SourceModel
import numpy as np

import collections

class ExtendedSource(SourceModel):
    def __init__(self,name,spatialModel,spectralModel):

        self.name                 = name
        self.functionName         = "Extended source"
    
    
        if(not callable(spatialModel)):
        
            raise RuntimeError("The provided spatial model must be callable")
    
        else:
        
            self.spatialModel      = spatialModel


        if(not callable(spectralModel)):
    
            raise RuntimeError("The provided spectral model must be callable")
    
        else:
        
            self.spectralModel      = spectralModel

        pass
        
        self.parameters           = collections.OrderedDict()
        
        for k,v in self.spatialModel.parameters.iteritems():
            
            self.parameters[ k ] = v
        
        
        #Parameters from the spectral model are already imported elsewhere
        
        #for k,v in self.spectralModel.parameters.iteritems():
            
        #    self.parameters[ k ] = v
        
        self.formula = self.spatialModel.formula
        
    pass

    def getFlux(self,energies):
        return self.spatialModel.integratedFlux(energies)*self.spectralModel(energies)

    def getBrightness(self,RA,Dec,energies):
        return self.spatialModel(RA,Dec,energies)*self.spectralModel(energies)
    
    def getBoundaries(self):
        
        return self.spatialModel.getBoundaries()



pass
