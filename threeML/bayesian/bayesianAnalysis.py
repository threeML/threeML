from threeML.parallel.ParallelClient import ParallelClient

from threeML.config.Config import threeML_config

from threeML.io.ProgressBar import ProgressBar

from threeML.io.triangle import corner

import emcee
import emcee.utils

import numpy
import time

def sampleWithProgress( p0, sampler, nsamples, **kwargs ):
    
    progress = ProgressBar( nsamples )
    
    for i, result in enumerate( sampler.sample(p0, iterations = nsamples, **kwargs) ):
        
        progress.animate( ( i + 1 ) )
        
        pos, prob, state = result
    
    progress.animate( nsamples )
    print("")
    return pos, prob, state

def sampleWithoutProgress( p0, sampler, nsamples, **kwargs ):
    
    return sampler.run_mcmc( p0, nsamples, **kwargs )

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
    
    def posterior( self, trialValues ):
        
        #Assign this trial values to the parameters and
        #store the corresponding values for the priors
        lps = []
        
        for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
            
            thisParam = self.likelihoodModel.parameters[ srcName ][ paramName ]
                  
            thisParam.setValue( trialValues[i] )
            
            pval = thisParam.getPriorValue()
            
            if not numpy.isfinite( pval ):
                
                return -numpy.inf
            
            lps.append( pval )
            
        logLike = numpy.sum( map( lambda dataset: dataset.getLogLike(), self.dataList.values() ) )
        
        if not numpy.isfinite( logLike ):
            
            return -numpy.inf
        
        logPrior = numpy.sum( lps )
        
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
        
        print("Running burn-in of %s samples..." % burn_in )

            
        pos, prob, state = sampleWithProgress( p0, sampler, burn_in )
        
        #Reset sampler
        
        sampler.reset()
        
        #Run the true sampling
        
        print("Sampling...")
        
        beg = time.time()
        
        _ = sampleWithoutProgress( pos, sampler, nsamples, rstate0=state ) 
        
        end = time.time()
        
        print("%s" %(end - beg))
        
        #sampler.run_mcmc( pos, nsamples, rstate0=state )
        
        acc = numpy.mean( sampler.acceptance_fraction )
        
        print( "Mean acceptance fraction: %s" % acc )
        
        self.sampler = sampler
        
        return self.getSamples()
    
    def getSamples( self ):
        
        return self.sampler.flatchain
    
    def cornerPlot( self ):
        
        if hasattr( self, "sampler" ):
            
            labels = []
            priors = []
            
            for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
                
                thisLabel = "%s of %s" % ( paramName, srcName )
                
                labels.append( thisLabel )
                
                priors.append( self.likelihoodModel.parameters[ srcName ][ paramName ].prior )
            
            fig = corner( self.sampler.flatchain, labels=labels, 
                          quantiles=[0.16, 0.50, 0.84],
                          priors = priors )
            
            return fig
        
        else:
            
            raise RuntimeError("You have to run the sampler first, using the sample() method")
