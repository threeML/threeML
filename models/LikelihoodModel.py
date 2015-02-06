from PointSource import PointSource
from ExtendedSource import ExtendedSource
import collections

class MyCollection(object):
  '''
  An ordered collections of key,value pairs which can be accessed
  by name or by numeral id
  '''
  def __init__(self):
    self.dict                 = collections.OrderedDict()
  
  def __getitem__(self,id_or_name):
    
    if(isinstance(id_or_name,int)):
      
      #by ID
      return self.dict.values()[id_or_name]
    
    elif(isinstance(id_or_name,str)):
      
      #By name
      return self.dict[id_or_name]
    
    else:
      raise RuntimeError("%s must be either an int or a str" %(id_or_name))
   
  def __setitem__(self,key,val):
     if(isinstance(key,int)):
       raise RuntimeError("Cannot use integers as key")
     
     self.dict[key]           = val
  
  def __len__(self):
     return len(self.dict.keys())
  
class LikelihoodModel(object):
  
  def __init__(self,*sources):
    
    #Store sources in ordered dictionaries, so that we can
    #access them by name or by id (ordinal number)
    
    self.pointSources         = MyCollection()
    self.extendedSources      = MyCollection()
    
    if(len(sources)==0):
      raise RuntimeError("You have to provide at least one source")
    
    #Loop through the provided sources
    for source in sources:
      
      if(isinstance(source,ExtendedSource)):
        
        #Current source is an Extended Source
        
        self.extendedSources[source.name] = source
      
      elif(isinstance(source,PointSource)):
        #Current source is a point source
        
        self.pointSources[source.name]    = source
      
      else:
      
        #User error, source is not a source (!)
        
        raise RuntimeError("One of the argument provided is not a "+ 
                           "point source nor an extended source")
      
      pass
      
    pass
  pass
  
  def getNumberOfPointSources(self):
    return len(self.pointSources)
  
  def getNumberOfExtendedSources(self):
    return len(self.extendedSources)
  
  def getPointSourcePosition(self,id_or_name):
    '''
    Return the position of the point source. You can use either the name
    or the ID of the source
    '''
    return self.pointSources[id_or_name].getPosition()
  pass
  
  def getPointSourceFluxes(self,id_or_name, energies):
    return self.pointSources[id_or_name].getFlux(energies)
  
  def getPointSourceName(self,id_or_name):
    return self.pointSources[id_or_name].name
  
  
  
  
    
