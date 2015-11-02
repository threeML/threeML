from threeML.parallel.ParallelClient import ParallelClient

from threeML.config.Config import threeML_config

from threeML.io.ProgressBar import ProgressBar

from threeML.io.triangle import corner

from threeML.exceptions.CustomExceptions import ModelAssertionViolation

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
    
    def _logp( self, trialValues ):
        
        #Compute the sum of the log-priors
        
        logp = 0
        
        for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
            
            thisParam = self.likelihoodModel.parameters[ srcName ][ paramName ]
            
            logp += thisParam.prior( trialValues[i] )
            
        return logp
        
    def _logLike( self, trialValues ):
        
        #Compute the log-likelihood
                
        for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
            
            thisParam = self.likelihoodModel.parameters[ srcName ][ paramName ]
            
            thisParam.setValue( trialValues[i] )
        
        try:    
            
            logLike = numpy.sum( map( lambda dataset: dataset.getLogLike(), self.dataList.values() ) )
        
        except ModelAssertionViolation:
            
            return -numpy.inf
        
        except:
            
            raise
        
        if not numpy.isfinite( logLike ):
            
            return -numpy.inf
        
        else:
            
            return logLike
    
    def posterior( self, trialValues ):
        
        #Here we don't use the self._logp nor the self._logLike to
        #avoid looping twice over the parameters (for speed)
        
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
    
    def _getStartingPoint( self, nwalkers, variance = 0.1 ):
        
        #Generate the starting points for the walkers by getting random
        #values for the parameters close to the current value
        
        #Fractional variance for randomization
        #(0.1 means var = 0.1 * value )
                
        p0 = [ ]
        
        for i in range( nwalkers ):
            
            thisP0 = []
            
            for (srcName, paramName) in self.freeParameters.keys():
                        
                thisPar = self.likelihoodModel.parameters[ srcName ][ paramName ]
                
                thisVal = thisPar.getRandomizedValue( variance )
                
                thisP0.append( thisVal )
            
            p0.append( thisP0 )
        
        return p0     
        
    
    def samplePT( self, ntemps, nwalkers, burn_in, nsamples ):
        '''
        Sample with parallel tempering
        '''
        
        self.freeParameters = self.likelihoodModel.getFreeParameters()
                
        ndim = len( self.freeParameters.keys() )
        
        sampler = emcee.PTSampler( ntemps, nwalkers, ndim, self._logLike, self._logp )
        
        #Get one starting point for each temperature
        
        p0 = numpy.empty( ( ntemps, nwalkers, ndim ) )
        
        for i in range( ntemps ):
            
            p0[i,:,:] = self._getStartingPoint( nwalkers )
        
        print("Running burn-in of %s samples...\n" % burn_in )
        
        p, lnprob, lnlike = sampleWithProgress( p0, sampler, burn_in )
        
        #Reset sampler
        
        sampler.reset()
        
        print("\nSampling...\n")
        
        p, lnprob, lnlike = sampleWithProgress( p, sampler, nsamples, 
                                                lnprob0 = lnprob, lnlike0 = lnlike )
        
        self.sampler = sampler
        self.samples = sampler.flatchain.reshape(-1, sampler.flatchain.shape[-1])
        
        return self.getSamples()
        
    
    def sample( self, nwalkers, burn_in, nsamples ):
        
        self.freeParameters = self.likelihoodModel.getFreeParameters()
                
        ndim = len( self.freeParameters.keys() )
        
        #Get starting point
        
        p0 = self._getStartingPoint( nwalkers )
        
        if threeML_config['parallel']['use-parallel']:
        
            c = ParallelClient()
            view = c[:]
        
            sampler = emcee.EnsembleSampler( nwalkers, ndim, 
                                             self.posterior,
                                             pool = view) 
        
        else:
            
            sampler = emcee.EnsembleSampler( nwalkers, ndim, 
                                             self.posterior )
        
        print("Running burn-in of %s samples...\n" % burn_in )
        
        pos, prob, state = sampleWithProgress( p0, sampler, burn_in )
        
        #Reset sampler
        
        sampler.reset()
        
        #Run the true sampling
        
        print("\nSampling...\n")
                
        _ = sampleWithProgress( pos, sampler, nsamples, rstate0=state ) 
        
        #sampler.run_mcmc( pos, nsamples, rstate0=state )
        
        acc = numpy.mean( sampler.acceptance_fraction )
        
        print( "Mean acceptance fraction: %s" % acc )
        
        self.sampler = sampler
        self.samples = sampler.flatchain
        
        return self.getSamples()
    
    def getSamples( self ):
        
        if hasattr( self, "samples" ):
            
            return self.samples
        
        else:
            
            raise RuntimeError("You have to run the sampler first, using the sample() method")
    
    def cornerPlot( self, **kwargs ):
        
        if hasattr( self, "samples" ):
            
            labels = []
            priors = []
            
            for i,( srcName, paramName ) in enumerate( self.freeParameters.keys() ):
                
                thisLabel = "%s of %s" % ( paramName, srcName )
                
                labels.append( thisLabel )
                
                priors.append( self.likelihoodModel.parameters[ srcName ][ paramName ].prior )
            
            fig = corner( self.samples, labels=labels, 
                          quantiles=[0.16, 0.50, 0.84],
                          priors = priors, **kwargs )
            
            return fig
        
        else:
            
            raise RuntimeError("You have to run the sampler first, using the sample() method")
