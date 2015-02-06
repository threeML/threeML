from threeML.bayesian import Priors

class Parameter(object):
  def __init__(self,name,initValue,minValue,maxValue,delta,**kwargs):
    self.name                 = str(name)
    self.value                = initValue
    self.minValue             = minValue
    self.maxValue             = maxValue
    self.delta                = delta
    self.unit                 = ''
    
    self.fixed                = False
    self.nuisance             = False
    self.dataset              = None
    self.callback             = []
    self.normalization        = False
    
    for k,v in kwargs.iteritems():
      if(k.lower()=='unit'):
        self.unit             = str(v)
      elif(k.lower()=='normalization'):
        self.normalization    = bool(v)
      elif(k.lower()=='fixed'):
        self.fixed            = bool(v)
      elif(k.lower()=='nuisance'):
        self.nuisance         = bool(v)
      elif(k.lower()=='dataset'):
        self.dataset          = v
      pass
    pass
    
    #Default prior is a uniform prior
    if(self.normalization):
      #This is a scale parameter
      self.setPrior(Priors.LogUniformPrior(self.minValue,self.maxValue))
    else:
      self.setPrior(Priors.UniformPrior(self.minValue,self.maxValue))
  pass
  
  def setCallback(self,callback):
    #The callback functions will be executed on any parameter value change
    self.callback.append(callback)
  pass
  
  def __repr__(self):
    if(self.fixed):
      ff                      = "fixed"
    else:
      ff                      = "free"
    pass
    
    return "%20s: %10g %10g %10g %10g %s %s" %(self.name,self.value,self.minValue,self.maxValue,self.delta,ff,self.unit)
  pass
  
  def setValue(self,value):
    self.value                = float(value)
    
    if(abs(self.delta) > 0.2*abs(self.value)):
      #Fix the delta to be less than 50% of the value
      self.delta              = 0.2 * self.value
    
    for c in self.callback:
      c()
  pass
  
  def setBounds(self,minValue,maxValue):
    self.minValue             = minValue
    self.maxValue             = maxValue
    self.prior.setBounds(minValue,maxValue)
  pass
  
  def setDelta(self,delta):
    self.delta                = delta
  pass
  
  def setPrior(self,prior):
    self.prior                = prior
  pass
  
  def setDataset(self,dataset):
    self.dataset              = dataset
  
  def fix(self):
    self.fixed                = True
  pass
  
  def free(self):
    self.fixed                = False
  pass
  
  def isNuisance(self):
    return self.nuisance
  
  def isNormalization(self):
    return self.normalization
  
  def isFixed(self):
    return self.fixed
  
  def isFree(self):
    return (not self.fixed)
pass
