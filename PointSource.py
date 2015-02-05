from SourceModel import SourceModel
import collections
from Parameter import Parameter

class PointSource(SourceModel):
  
  def __init__(self,name,ra,dec,spectralModel):
    
    #TODO: accept input coordinates in other systems and convert them immediately
    #to J2000 Equatorial R.A. and Dec.
    
    self.name                 = name
    self.functionName         = "Point source"
    self.formula              = r"\begin{equation}f(RA',Dec') = \delta(RA'-RA)\delta(Dec'-Dec)\end{equation}"
    self.parameters           = collections.OrderedDict()
    self.parameters['RA']     = Parameter('RA', ra, 0.0, 360.0, 0.01,fixed=True,nuisance=False,dataset=None,unit='deg')
    self.parameters['Dec']    = Parameter('Dec', dec, -90.0, 90.0, 0.01,fixed=True,nuisance=False,dataset=None,unit='deg')
    
    if(not callable(spectralModel)):
      
      raise RuntimeError("The provided spectral model must be callable")
    
    else:
    
      self.spectralModel      = spectralModel
    
    pass
  pass
  
  def getFlux(self,energies):
    return self.spectralModel(energies)
  
  def getPosition(self):
    return (self.parameters['RA'].value, self.parameters['Dec'].value)
  
pass

