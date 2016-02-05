import os, sys
import pkg_resources
import yaml


class Config( object ):
    
    def __init__( self ):
        
        # Read the config file
        
        # Define a list of possible path where the config file might be
        # The first successful path will be the active configuration file
        
        possiblePaths = []
        
        # First possible path is the .threeML directory under the user-home
        
        possiblePaths.append( os.path.join( os.path.expanduser( '~' ), '.threeML'  ) )
        
        # Second possible path is the config subdir under the package path
        # (which is where this config.py file is)

        distribution = pkg_resources.get_distribution("threeML")

        possiblePaths.append( os.path.join( distribution.location, 'threeML/config' ) )

        self._configuration = None
        self._filename = None
        
        for path in possiblePaths:
            
            thisFilename = os.path.join( path, 'threeML_config.yml' )
            
            if os.path.exists( thisFilename ):
                
                with open( thisFilename ) as f:
                
                    self._configuration = yaml.safe_load( f )
                
                print("Configuration read from %s" %( thisFilename ) )
                
                self._filename = thisFilename
                
                break
            
            else:
                
                continue
        
        if self._configuration is None:
            
            raise RuntimeError("Could not find threeML_config.yml in any of %s" %( possiblePaths ))

    
    def __getitem__(self, key):
        
        if key in self._configuration.keys():
            
            return self._configuration[ key ]
        
        else:
            
            raise RuntimeError("Configuration key %s does not exist in %s." %( key, self._filename ) )
    
    def __repr__(self):
        
        return yaml.dump( self._configuration, default_flow_style=False )


# Now read the config file, so it will be available as Config.c
threeML_config = Config()

