from threeML.parallel.ParallelClient import ParallelClient

from threeML.config.Config import threeML_config


import emcee
import emcee.utils

class bayesianAnalysis( object ):
    
    def __init__(self, likelihoodModel, dataList, **kwargs):
    
        #Process optional keyword parameters
        self.verbose             = False
            
        for k,v in kwargs.iteritems():
          
          if(k.lower()=="verbose"):
          
            self.verbose           = bool(kwargs["verbose"])
        
            
        self.likelihoodModel      = likelihoodModel
            
        self.dataList             = dataList
        
        for dataset in self.dataList.values():
          
          dataset.setModel(self.likelihoodModel)
    
    def posterior( self, trialList ):
        
        for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
                        
            self.likelihoodModel.parameters[ srcName ][ paramName ].setValue( trialValues[i] )
        
        logLike = numpy.sum( map( lambda dataset: dataset.getLogLike(), self.dataList ) )
        
        logPrior = numpy.sum( map( lambda val, parameter: parameter.prior( val ), zip(  ) ) )
        
        return logLike + logPrior
    
    def sample( self, nwalkers, burn_in, nsamples ):
        
        self.freeParameters = self.likelihoodModel.getFreeParameters()
                
        ndim = len( self.freeParameters.keys() )
                
        #Generate the starting points for the walkers by getting random
        #values for the parameters close to the current value
        
        #Fractional variance for randomization
        #(0.1 means var = 0.1 * value )
        variance = 0.1
        
        p0 = [ ]
        
        for i in range( nwalkers ):
            
            thisP0 = []
            
            for (srcName, paramName) in self.freeParameters.keys():
                        
                thisPar = self.likelihoodModel.parameters[ srcName ][ paramName ]
                
                thisVal = thisPar.getRandomizedValue( variance )
                
                thisP0.append( thisVal )
            
            p0.append( thisP0 )
        
        if threeML_config['parallel']['use-parallel']:
        
            c = ParallelClient()
            view = c[:]
        
            sampler = emcee.EnsembleSampler( nwalkers, ndim, 
                                             self.posterior,
                                             pool = view) 
        
        else:
            
            sampler = emcee.EnsembleSampler( nwalkers, ndim, 
                                             self.posterior )
        
        pos, prob, state = sampler.run_mcmc( p0, burn_in )
        
        sampler.reset()
        
        sampler.run_mcmc( pos, nsamples, rstate0=state )
        
        acc = numpy.mean( sampler.acceptance_fraction )
        
        print( "Mean acceptance fraction: " % acc )
        
        return sampler
