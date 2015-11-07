import os

def fileExistingAndReadable( filename ):
    
    if os.path.exists( filename ):
        
	#Try to open it
	
	try:
	    
	    with open( filename ):
	    
	        pass
	
	except:
	    
	    return False
	
	else:
	
	    return True

    else:
        
	return False


def sanitizeFilename( filename ):
    
    return os.path.expandvars( os.path.expanduser( filename ) )
