import collections
from Parameter import Parameter
import fancyDisplay
from IPython.display import display, Latex, HTML

class SourceModel(object):  
  def __getitem__(self,argument):
    return self.parameters[argument]
  
  def __repr__(self):
    print("Spatial model: %s" %(self.functionName))
    print("Formula:\n")
    display(Latex(self.formula))
    print("")
    print("Current parameters:\n")
    table                    = fancyDisplay.HtmlTable(7)
    table.addHeadings("Name","Value","Minimum","Maximum","Delta","Status","Unit")
    for k,v in self.parameters.iteritems():
      if(v.isFree()):
        ff                   = "free"
      else:
        ff                   = "fixed"
      pass
      table.addRow(v.name,v.value,v.minValue,v.maxValue,v.delta,ff,v.unit)
    pass
    display(HTML(table.__repr__()))
    
    return ''
  pass
pass
