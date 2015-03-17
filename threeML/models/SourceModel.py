import collections
from Parameter import Parameter

from astropy.table import Table

from IPython.display import display, Latex, HTML

class SourceModel(object):  
  
  def __init__(self):
    print("HEY")
  
  def __getitem__(self,argument):
    return self.parameters[argument]
  
  def getAllParameters(self):
  
    allParameters                  = collections.OrderedDict()
    
    for dic in [self.parameters,self.spectralModel.parameters]:
      
      for k,v in dic.iteritems():
        
        allParameters[k]           = v
    
    return allParameters 
    
  
  def __repr__(self):
    
    print("Spatial model: %s" %(self.functionName))
    print("Formula:\n")
    
    display(Latex(self.formula))
    
    print("")
    print("Current parameters:\n")
    
    data = []
    nameLength = 0
        
    for dic in [self.parameters,self.spectralModel.parameters]:
      for k,v in dic.iteritems():
        if(v.isFree()):
          ff                   = "free"
        else:
          ff                   = "fixed"
        pass
        data.append([v.name,v.value,v.minValue,v.maxValue,v.delta,ff,v.unit])
        
        if(len(v.name) > nameLength):
          nameLength = len(v.name)
      
      pass
    pass
    
    table                     = Table(rows = data,
                                      names = ["Name","Value","Minimum","Maximum","Delta","Status","Unit"],
                                      dtype=('S%i' %nameLength, float, float, float, float, "S5",'S15'))
    
    display(table)
        
    return ''
  pass
  
pass
