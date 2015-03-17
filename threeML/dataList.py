#Author: G.Vianello (giacomov@stanford.edu)

import collections

class DataList(collections.OrderedDict):
  '''
  A container for datasets. Can be accessed as a dictionary,
  with the [key] operator.
  '''
  def __init__(self, *datasets):
    
    #Enforce the uniqueness of the names for the datasets
    names                     = map(lambda x:x.getName(),datasets)
    
    #sets contains by definition only unique elements
    uniqueNames               = set(names)
    
    if( len(names) != len(uniqueNames) ):
    
      raise RuntimeError("Duplicated names for datasets. You have to use"+
                         " a unique name for each dataset.")
    
    #Build the parent class
    super(DataList, self).__init__()
    
    
    #Store the datasets
    
    for ds in datasets:
      
      self[ds.getName()]           = ds       
pass
