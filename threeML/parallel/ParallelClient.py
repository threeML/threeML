import dill
from IPython.parallel import Client

class ParallelClient(Client):
  
  def __init__(self, *args, **kwargs):
    
    #Just a wrapper around the IPython Client class
    #forcing the use of dill for object serialization
    #(more robust, and allows for serialization of class
    #methods)
    super(ParallelClient, self).__init__(*args, **kwargs)
    
    #This will propagate the use_dill to all running
    #engines    
    self.direct_view().use_dill()
