from threeML.pluginPrototype import pluginPrototype

from threeML.models.Parameter import Parameter

from threeML.io.fileUtils import fileExistingAndReadable, sanitizeFilename

from threeML.pyModelInterface import pyToCppModelInterface

from hawc import liff

import os, sys, collections

defaultMinChannel = 0
defaultMaxChannel = 9

__instrument_name = "HAWC"

class HAWCLike( pluginPrototype ):
    
    def __init__( self, name, maptree, response, ntransits ):
        
        self.name = str( name )
        
        #Sanitize files in input (expand variables and so on)
        
        self.maptree = os.path.abspath( sanitizeFilename( maptree ) )
        
        self.response = os.path.abspath( sanitizeFilename( response ) )
        
        #
        self.ntransits = ntransits
        
        #Check that they exists and can be read
        
        if not fileExistingAndReadable( self.maptree ):
        
            raise IOError("MapTree %s does not exist or is not readable" % maptree)
    
        if not fileExistingAndReadable( self.response ):
        
            raise IOError("Response %s does not exist or is not readable" % response)
        
        #Post-pone the creation of the LIFF instance to when
        #we have the likelihood model
        
        self.instanced = False
        
        #Default value for minChannel and maxChannel
        
        self.minChannel = int( defaultMinChannel )
        self.maxChannel = int( defaultMaxChannel )
        
        #By default the fit of the CommonNorm is deactivated
        
        self.deactivateCommonNorm()
        
        #Further setup
        
        self.__setup()
    
    def __setup(self):
        
        #I put this here so I can use it both from the __init__ both from
        #the __setstate__ methods
        
        #Create the dictionary of nuisance parameters
        
        self.nuisanceParameters = collections.OrderedDict()
        self.nuisanceParameters['CommonNorm'] = Parameter("CommonNorm",1.0,0.5,1.5,0.01,
                                                           fixed=True,nuisance=True)
    
    def __getstate__(self):
        
        #This method is used by pickle before attempting to pickle the class
        
        #Return only the objects needed to recreate the class
        #IN particular, we do NOT return the theLikeHAWC class,
        #which is not pickeable. It will instead be recreated
        #on the other side
        
        d = {}
        
        d['name']= self.name
        d['maptree'] = self.maptree
        d['response'] = self.response
        d['ntransits'] = self.ntransits
        d['model'] = self.model
        d['minChannel'] = self.minChannel
        d['maxChannel'] = self.maxChannel
        
        return d
    
    def __setstate__( self, state ):
        
        #This is used by pickle to recreate the class on the remote
        #side
        name = state['name']
        maptree = state['maptree']
        response = state['response']
        ntransits = state['ntransits']
                
        self.__init__( name, maptree, response, ntransits )
        
        #Now report the class to its state
        
        self.setActiveMeasurements( state['minChannel'], state['maxChannel'] )
        
        self.setModel( state['model'] )

    
    def setActiveMeasurements( self, minChannel, maxChannel ):
        
        self.minChannel = int( minChannel )
        self.maxChannel = int( maxChannel )
        
        if self.instanced:
            
            sys.stderr.write("Since the plugins was already used before, the change in active measurements" +
                             "will not be effective until you create a new JointLikelihood or Bayesian" +
                             "instance")
        
    def setModel( self, LikelihoodModelInstance ):
        '''
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        '''
        
        #Instance the python - C++ bridge
        self.model = LikelihoodModelInstance
        
        self.pymodel = pyToCppModelInterface( self.model )
        
        #Now init the HAWC LIFF software
    
        try:
        
            self.theLikeHAWC = liff.LikeHAWC( self.maptree, 
                                              self.ntransits,
                                              self.response,
                                              self.pymodel,
                                              self.minChannel,
                                              self.maxChannel )
        
        except:
            
            print("Could not instance the LikeHAWC class from LIFF. " +
                               "Check that HAWC software is working")
            
            raise
        
        else:
            
            self.instanced = True
                    
        #Now set a callback in the CommonNorm parameter, so that if the user or the fit
        #engine or the Bayesian sampler change the CommonNorm value, the change will be
        #propagated to the LikeHAWC instance
        
        self.nuisanceParameters['CommonNorm'].setCallback( self._CommonNormCallback )
        
    
    def _CommonNormCallback( self ):
        
        self.theLikeHAWC.SetCommonNorm( self.nuisanceParameters['CommonNorm'].value )
    
    def getName(self):
        '''
        Return a name for this dataset (likely set during the constructor)
        '''
        return self.name
    
    def activateCommonNorm( self ):
        
        self.fitCommonNorm = True
    
    def deactivateCommonNorm( self ):
        
        self.fitCommonNorm = False
    
    def getLogLike(self):
        
        '''
        Return the value of the log-likelihood with the current values for the
        parameters
        '''
        
        self.pymodel.update()
        
        logL = self.theLikeHAWC.getLogLike( self.fitCommonNorm )
                
        return logL
  
    def getNuisanceParameters(self):
        '''
        Return a list of nuisance parameters. Return an empty list if there
        are no nuisance parameters
        '''
        
        return self.nuisanceParameters.keys()
  
    def innerFit(self):
        
        self.theLikeHAWC.SetBackgroundNormFree( self.fitCommonNorm )
        
        self.pymodel.update()
        
        logL = self.theLikeHAWC.getLogLike( self.fitCommonNorm )
        
        self.nuisanceParameters['CommonNorm'].setValue( self.theLikeHAWC.CommonNorm() )
        
        return logL

