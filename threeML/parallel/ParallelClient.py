import dill
import yaml
import os

from threeML.config.Config import threeML_config

try:
    
    from IPython.parallel import Client

except:
    
    from ipyparallel import Client
    
    
from contextlib import contextmanager

@contextmanager
def parallel_computation( profile=None ):
    
    #Memorize the state of the use-parallel config
    old_state = bool( threeML_config['parallel']['use-parallel'] )
    
    old_profile = str( threeML_config['parallel']['IPython profile name'] )
    
    #Set the use-parallel feature on
    threeML_config['parallel']['use-parallel'] = True
    
    #Now use the specified profile (if any), otherwise the default one
    if profile is not None:
        
        threeML_config['parallel']['IPython profile name'] = str( profile )
    
    #Here is where the content of the with parallel_computation statement gets
    #executed
    try:
    
        yield
    
    finally:
        
        #This gets executed in any case, even if there is an exception
        
        #Revert back
        threeML_config['parallel']['use-parallel'] = old_state
        
        threeML_config['parallel']['IPython profile name'] = old_profile

class ParallelClient(Client):
  
  def __init__(self, *args, **kwargs):
    
    #Just a wrapper around the IPython Client class
    #forcing the use of dill for object serialization
    #(more robust, and allows for serialization of class
    #methods)
    
    if 'profile' not in kwargs.keys():
        
        kwargs['profile'] = threeML_config['parallel']['IPython profile name']
    
    super(ParallelClient, self).__init__(*args, **kwargs)
    
    #This will propagate the use_dill to all running
    #engines    
    self.direct_view().use_dill()
 
  def getNumberOfEngines( self ):
    
    return len( self.direct_view() )
