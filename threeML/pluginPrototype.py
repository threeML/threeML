'''
Define the interface for a plugin class.
'''

import abc

class pluginPrototype(object):
  __metaclass__               = abc.ABCMeta  
  
  @abc.abstractmethod
  def setModel(self,ModelManagerInstance):
    '''
    Set the model to be used in the joint minimization. Must be a ModelManager instance.
    '''
    pass
  pass
  
  @abc.abstractmethod
  def getName(self):
    '''
    Return a name for this dataset (likely set during the constructor)
    '''
    pass
  pass
    
  @abc.abstractmethod
  def getLogLike(self):
    '''
    Return the value of the log-likelihood with the current values for the
    parameters
    '''
    pass
  pass
  
  @abc.abstractmethod
  def getNuisanceParameters(self):
    '''
    Return a list of nuisance parameters. Return an empty list if there
    are no nuisance parameters
    '''
    pass
  pass
  
  @abc.abstractmethod
  def innerFit(self):
    '''
    This is used for the profile likelihood. Keeping fixed all parameters in the
    modelManager, this method minimize the logLike over the remaining nuisance
    parameters, i.e., the parameters belonging only to the model for this
    particular detector. If there are no nuisance parameters, simply return the
    logLike value.
    '''
    pass
  pass
pass
