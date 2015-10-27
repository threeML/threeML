import numpy
import abc
import math

import scipy.stats

class Prior(object):
  __metaclass__               = abc.ABCMeta
    
  @abc.abstractmethod
  def setBounds(self,newMinValue,newMaxValue):
    '''
    Set the minimum and maximum values for the parameter
    '''
    pass
  pass
  
  @abc.abstractmethod
  def getName(self):
    '''
    Return the name of the prior
    '''
    pass
  pass 
  
  @abc.abstractmethod
  def __call__(self,value):
    '''
    Return the logarithm of the prior pdf at the given value
    '''
    pass
  pass
  
class UniformPrior(Prior):
  def __init__(self,minValue,maxValue):
    self.minValue             = minValue
    self.maxValue             = maxValue
  pass
  
  def getName(self):
    return "UniformPrior"
  
  def setBounds(self,newMinValue,newMaxValue):
    self.minValue             = newMinValue
    self.maxValue             = newMaxValue
  pass
  
  def __call__(self,value):
    if(self.minValue < value < self.maxValue):
      return 0.0
    else:
      return -numpy.inf
    pass
  pass
  
  def multinestCall(self,cube):
    return cube * (self.maxValue - self.minValue ) + self.minValue
pass

class LogUniformPrior(Prior):
  def __init__(self,minValue,maxValue):
    self.minValue             = minValue
    self.maxValue             = maxValue
  pass
  
  def getName(self):
    return "LogUniformPrior"
  
  def setBounds(self,newMinValue,newMaxValue):
    self.minValue             = newMinValue
    self.maxValue             = newMaxValue
  pass
  
  def __call__(self,value):
    if(self.minValue < value < self.maxValue and value>0):
      #This is = log(1/value)
      return -math.log10(value)
    else:
      return -numpy.inf
    pass
  pass
  
  def multinestCall(self,cube):
    decades                   = math.log1010(self.maxValue)-math.log1010(self.minValue)
    startDecade               = math.log1010(self.minValue)
    return 10**((cube * decades) + startDecade)

pass

class GaussianPrior(Prior):
  
  def __init__(self, mu, sigma):
    
    self.mu = float(mu)
    self.sigma = abs( float(sigma) )
        
    
  def getName(self):
    
    return "GaussianPrior"
  
  def setBounds(self,newMinValue,newMaxValue):
    #Useless in this context
    pass
  
  def __call__(self, value):
    
    #Here I multiply by sigma because of scipy "scale" behavior
    #(see scipy documentation)
    
    val                       = scipy.stats.norm.pdf( value, loc=self.mu, scale=self.sigma ) * self.sigma
  
    if( val < 1e-15):
      
      return - numpy.inf
    
    else:
      
      return math.log10(val)
