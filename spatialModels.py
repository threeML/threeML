import collections
from Parameter import Parameter
import fancyDisplay
from IPython.display import display, Latex, HTML

class SpatialModel(object):  
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

class PointSource(SpatialModel):
  def __init__(self,ra,dec):
    self.functionName         = "Point source"
    self.formula              = r"\begin{equation}f(RA',Dec') = \delta(RA'-RA)\delta(Dec'-Dec)\end{equation}"
    self.parameters           = collections.OrderedDict()
    self.parameters['RA']     = Parameter('RA',ra,0.0,360.0,0.01,fixed=True,nuisance=False,dataset=None,unit='deg')
    self.parameters['Dec']    = Parameter('Dec',dec,-90.0,90.0,0.01,fixed=True,nuisance=False,dataset=None,unit='deg')
  pass
  
  def getRA(self):
    return self.parameters['RA'].value
  
  def getDec(self):
    return self.parameters['Dec'].value
  
pass
