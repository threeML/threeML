#Author: G.Vianello (giacomov@stanford.edu)

import collections

class DataList(object):
  '''
  A container for datasets. Can be accessed as a dictionary,
  with the [key] operator.
  '''
  
  def __init__(self, *datasets):
    
    self._innerDict = collections.OrderedDict()
    
    for d in datasets:
      
      if( d.getName() in self._innerDict.keys() ):
        
        raise RuntimeError("You have to use unique names for datasets. %s already exists." %(d.getName()))
      
      else:
      
        self._innerDict[ d.getName() ] = d 
  
  def __setitem__(self, key, value):
    
    #Enforce the unique name
    if key in self.keys():
    
      raise RuntimeError("You have to use unique names for datasets. %s already exists." %(key))
    
    else:
      
      self._innerDict[key] = value
  
  def __getitem__(self, key):
    
    return self._innerDict[key]
  
  def keys(self):
    
    return self._innerDict.keys()
  
  def values(self):
    
    return self._innerDict.values()
  
pass
