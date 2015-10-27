from threeML.bayesian import Priors

import numpy

import scipy.stats

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

  def __eq__(self,value):
    self.setValue(value)
    
  def __repr__(self):
    if(self.fixed):
      ff                      = "fixed"
    else:
      ff                      = "free"
    pass
    
    return "%20s: %10g %10g %10g %10g %s %s" %(self.name,self.value,self.minValue,self.maxValue,self.delta,ff,self.unit)
  pass
  
  def getRandomizedValue( self, var = 0.1 ):
    
    #Get a value close to the current value, but not identical
    #(used for the inizialization of Bayesian samplers)
    
    if ( self.minValue is not None ) or ( self.maxValue is not None ):
        
        #Bounded parameter. Use a truncated normal so we are guaranteed
        #to have a random value within the boundaries
        
        std = abs( var * self.value )
        
        if self.minValue is not None:
            
            a = ( self.minValue - self.value ) / std 
        
        else:
            
            a = - numpy.inf
        
        if self.maxValue is not None:
            
            b = ( self.maxValue - self.value ) / std 
        
        else:
            
            b = numpy.inf
              
        sample = scipy.stats.truncnorm.rvs( a, b, loc = self.value, scale = std, size = 1)
        
        if sample < self.minValue or sample > self.maxValue:
            
            raise RuntimeError("BUG!!")
        
        return sample[0]
    
    else:
        
        #The parameter has no boundaries
        
        return numpy.random.normal( self.value, var * self.value )
  
  def getValue(self):
    
    return self.value
  
  def getPriorValue( self ):
    
    return self.prior( self.value )
  
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


class SpatialParameter(object):
    
    #this class provides a place holder for spatial parameters that vary with energy, for the moment works exactly as the regular parameter with value indepedent from energy
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
        
    def __eq__(self,value):
        self.setValue(value)
    
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
    
    def getValue(self,energy):
        return self.value
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

