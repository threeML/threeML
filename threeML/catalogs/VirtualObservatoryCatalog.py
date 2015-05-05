import warnings

from astropy.vo.client.vos_catalog import VOSCatalog
from astropy.vo.client import conesearch
from astropy.vo.client.exceptions import VOSError


class VirtualObservatoryCatalog(object):
    
    def __init__(self, name, url, description):
                
        self.catalog = VOSCatalog.create(name, url, description=description)
    
    def query(self, skycoord, radius):     
        
        with warnings.catch_warnings():
            
            #Ignore all warnings, which are many from the conesearch module
            
            warnings.simplefilter('ignore')
            
            try:
                
                votable = conesearch.conesearch(skycoord, radius, 
                                                catalog_db=self.catalog,
                                                verb=3, verbose=True,
                                                cache=False)
            
            except VOSError as exc:
                
                print(exc.message)
                return None
        
        out = self.applyFormat(votable)
        
        #This is needed to avoid strange errors
        del votable
        
        return out
    
    def applyFormat(self):
        
        raise NotImplementedError("You have to re-implement this!")
