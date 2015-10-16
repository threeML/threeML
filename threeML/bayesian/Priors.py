import numpy
import abc
import math

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
      return -math.log(value)
    else:
      return -numpy.inf
    pass
  pass
  
  def multinestCall(self,cube):
    decades                   = math.log10(self.maxValue)-math.log10(self.minValue)
    startDecade               = math.log10(self.minValue)
    return 10**((cube * decades) + startDecade)

pass

class GaussianPrior(Prior):
  
  def __init__(self, mu, sigma):
    
    self.mu                   = float(mu)
    self.sigma                = float(sigma)
    self.two_sigmasq          = 2 * self.sigma**2.0
    self.one_on_sigmaSqrt_2pi = 1.0 / ( self.sigma * math.sqrt(2 * math.pi) )
    
  def getName(self):
    
    return "GaussianPrior"
  
  def setBounds(self,newMinValue,newMaxValue):
    #Useless in this context
    pass
  
  def __call__(self, value):
    
    val                       = (   self.one_on_sigmaSqrt_2pi 
                                  * math.exp( - (value - self.mu)**2 
                                                      / 
                                              self.two_sigmasq ) )
  
    if( val < 1e-15):
      
      return - numpy.inf
    
    else:
      
      return math.log(val)
