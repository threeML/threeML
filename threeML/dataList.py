import collections

class dataList(object):
  def __init__(self,*datasets):
    
    #Enforce the uniqueness of the names for the datasets
    names                     = map(lambda x:x.getName(),datasets)
    if(len(names)!=len(set(names))):
      raise RuntimeError("Duplicated names for datasets. You have to use a unique name for each dataset.")
    pass

    #Save in a ordered dictionary (which keeps the order of the elements)
    self.datasets             = collections.OrderedDict()  
    for ds in datasets:
      self.datasets[ds.getName()] = ds
    pass        
  pass
  
pass
