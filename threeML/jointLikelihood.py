from threeML.minimizer import minimization
import collections

import numpy
import scipy.optimize
import scipy.stats
import sys
import matplotlib.pyplot as plt

class JointLikelihood(object):
  def __init__(self,modelManager,**kwargs):
    
    #Process optional keyword parameters
    self.verbose             = False
    defaultMinimizer         = "MINUIT"
    
    for k,v in kwargs.iteritems():
      if(k.lower()=="verbose"):
        self.verbose           = bool(kwargs["verbose"])
      elif(k.lower()=="minimizer"):
        defaultMinimizer       = v.upper()
      pass
    pass
    
    self.modelManager         = modelManager
    
    self.dataSets             = modelManager.dataList.datasets.values()
    for ds in self.dataSets:      
      #The following is to ensure the proper set of some
      #datasets (for example, for the LAT datasets this
      #generate the new XML model which allow the user to set
      #a prior for the effective area correction after instanciating
      #this class)
      ds.setModel(self.modelManager)
      dumb                   = ds.getLogLike()
    pass
    
    self._buildGlobalLikelihoodFunctions()
    
    self.sampler             = None
    
    
    #These will store the best fit results
    self.bestFitValues        = collections.OrderedDict()
    self.approxErrors         = collections.OrderedDict()
    
    #Default minimizer is MINUIT
    self.setMinimizer(defaultMinimizer)
  pass
  
  def _buildGlobalLikelihoodFunctions(self):
    self.ncalls                   = 0
    
    #Global likelihood function, profiling out nuisance parameters
    def minusLogLikeProfile(args):
      self.ncalls                += 1
      #Assign the new values to the parameters
      for i,parname in enumerate(self.freeParameters.keys()):
        self.modelManager[parname].setValue(args[i])
      pass
      
      valuesString                = self.modelManager.printParamValues(False)      
      
      #Now profile out nuisance parameters and compute the new value 
      #for the likelihood
      globalLogLike              = 0
      for dataset in self.dataSets:
          #print("Dataset %s" % dataset.getName())
          dataset.innerFit()
          #print("Inner fit done")
          globalLogLike         += dataset.getLogLike()
          #print("Like computation done")
      pass
      
      if("%s" % globalLogLike=='nan'):
        print("Warning: these parameters returned a logLike = Nan: %s" %(valuesString))
        return 1e6
      
      if(self.verbose):
        print("Trying with parameters %s, resulting in logL = %s" %(valuesString,globalLogLike))
      
      return globalLogLike*(-1)
    pass
          
    #Global likelihood function
    def minusLogLike(args):
    
      #Assign the new values to the parameters of the model
      values                  = []
      for i,par in enumerate(self.freeParameters.keys()):
        self.modelManager[par].setValue(args[i])
        values.append(args[i])
            
      #Now compute the new value for the likelihood
      globalLogLike              = 0
      for dataset in self.dataSets:
          globalLogLike         += dataset.getLogLike()
      pass
      
      if(self.verbose):
        print("Trying with parameters %s, resulting in logL = %s" %(",".join(map(lambda x:str(x),values)),globalLogLike))
      
      return globalLogLike*(-1)
    pass
        
    #Store it
    self.minusLogLike         = minusLogLike
    self.minusLogLikeProfile  = minusLogLikeProfile
  pass
  
  def explore(self,nwalkers,nsamplesPerWalker,burn=None):
    
    import emcee
    
    self.freeParameters       = self.modelManager.getFreeParameters()
    
    #Default burnout is nsamples/10:
    if(burn==None):
      burn                    = int(nsamplesPerWalker/10.0)
      print("Using default burn of nsamples/10 = %i" %(burn))
    pass
    
    def lnprior(pars):
      globalLnPrior           = 0
      for i,p in enumerate(self.freeParameters.keys()):
        lnprior               = self.modelManager[p].prior
        #value                 = self.modelManager.setParameterValue(p,pars[i])
        globalLnPrior        += lnprior(pars[i])
        #print("Parameter %s = %s -> lnprior = %s" %(p,pars[i],lnprior(pars[i])))
      pass
            
      #print("globalLnPrior is %s\n" %(globalLnPrior))
      return globalLnPrior
    
    def lnprob(theta):
      lp                      = lnprior(theta)
      if not numpy.isfinite(lp):
        #print("lnprob is infinite\n")
        return -numpy.inf
      tot                     = lp + self.minusLogLike(theta)*(-1)
      #print("%s" %(tot-lp))
      return tot
    
    def lnprob2(theta):
      return self.minusLogLike(theta)*(-1)
    
    #Get some init values from the profile likelihood fit
    if(len(self.freeParameters.keys()) < 20):
      print("Performing profile-likelihood minimization to get init values...")
      res                       = self.fit()
      print("\nNow sampling posterior distribution with MCMC...")
      
      #This merges the two lists
      allValues                 = res[0].values()
    else:
      res                       = self.fit(False,True)
      print res
      allValues                 = map(lambda x:x.value,self.freeParameters.values())
      print allValues
      print self.minusLogLike(allValues)
    pass
    
    ntemps                    = 20
    
    ndim                      = len(allValues)
    #p0                        = [numpy.array(allValues)*numpy.random.uniform(0.9,1.1,ndim) for i in range(nwalkers)]
    p0                        = numpy.random.uniform(0.9,1.1,size=(ntemps,nwalkers,ndim))*numpy.array(allValues)
    
    self.sampler              = emcee.PTSampler(ntemps,nwalkers, ndim, lnprob2,lnprior)
    #self.sampler              = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    self.sampler.reset()
    if(burn>0):
      for p,lnprob,lnlike in self.sampler.sample(p0,iterations=burn):
        pass
      #r                         = self.sampler.run_mcmc(p0, burn)
      self.sampler.reset()
    else:
      p                       = p0
    pass
    
    for p, lnprob,lnlike in self.sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike,iterations=nsamplesPerWalker):
      pass
    
    #r                         = self.sampler.run_mcmc(p0, nsamplesPerWalker) 
    print("done")   
    
    ndim                      = self.sampler.chain.shape[-1]
    self.samples              = self.sampler.chain[:,:, :, :].reshape((-1, ndim))
  pass

  def multinest(self,*args,**kwargs):
    import pymultinest
    
    #res                       = self.fit(False,True)
    
    #f                         = open("calls.txt","w+")
    
    self.freeParameters       = self.modelManager.getFreeParameters()
        
    def prior(cube, ndim, nparams):
      for i,p in enumerate(self.freeParameters.values()):
        cube[i]               = p.prior.multinestCall(cube[i]) 
      pass
    pass
        
    def loglike(cube, ndim, nparams):
      logL                    = self.minusLogLike(cube)*(-1)
      if(numpy.isnan(logL)):
        logL                  = -1e10
      #f.write(" ".join(map(lambda x:"%s" %x,cube[:ndim])))
      #f.write(" %s\n" % logL)
      return logL
    pass
    
    if('verbose' not in kwargs):
      kwargs['verbose']       = True
    if('resume' not in kwargs):
      kwargs['resume']        = False
    if('outputfiles_basename' not in kwargs):
      kwargs['outputfiles_basename'] = '_1_'
    pass
    kwargs['log_zero']        = -1e9
    pymultinest.run(loglike, prior, len(self.freeParameters), *args, **kwargs)
    print("done")
    
    #Collect the samples
    analyzer                   = pymultinest.Analyzer(n_params=len(self.freeParameters),outputfiles_basename=kwargs['outputfiles_basename'])
    
    eqw                        = analyzer.get_equal_weighted_posterior()
    self.samples               = eqw[:,:-1]
    self.posteriors            = eqw[:,-1]
    #f.close()
  pass

  
  def getPercentiles(self,burnout=0,**kwargs):
    '''
    Get percentiles from the current MCMC chain
    '''
    #Process optional parameters
    printout                  = True
    levels                    = [50,16,84]
    for k,v in kwargs.iteritems():
      if(k.lower()=="printout"):
        printout              = bool(v)
      elif(k.lower()=="levels"):
        levels                = list(v)
      pass
    pass
        
    parnames                  = self.freeParameters.keys()
    percentiles               = collections.OrderedDict()
    for i,p in enumerate(parnames):
      percentiles[p]          = numpy.percentile(self.samples[:,i],levels)
    pass
    
    if(printout):
      print("Percentiles: %s" %(levels))
      for k,v in percentiles.iteritems():
        print("%-40s = %.4g %.4g %.4g" %(k,v[0],v[1]-v[0],v[2]-v[0]))
      pass
    pass
    
    return percentiles
  pass
  
  def setMinimizer(self,minimizer):
    if(minimizer.upper()=="MINUIT"):
      self.Minimizer          = minimization.MinuitMinimizer
    elif(minimizer.upper()=="SCIPY"):
      self.Minimizer          = minimization.ScipyMinimizer
    elif(minimizer.upper()=="BOBYQA"):
      self.Minimizer          = minimization.BOBYQAMinimizer
    else:
      raise ValueError("Do not know minimizer %s" %(minimizer))
    pass
  pass
  
  def fit(self,minos=False,normOnly=False):
    
    if(1==0):
      #Fit the normalizations of the spectral model first, otherwise, if they are too far off, they will
      #prevent the minimizer to find a solution
      self.freeParameters       = self.modelManager.getFreeNormalizationParameters()
      if(len(self.freeParameters.values())>0):
        minimizer               = self.Minimizer(self.minusLogLikeProfile,self.freeParameters)
        xs,xserr,logLmin        = minimizer.minimize(False,False)
      pass
      
      if(normOnly):
        self.freeParameters       = self.modelManager.getFreeParameters()
        return self.modelManager.getFreeNormalizationParameters(),logLmin
     
      #Now, assuming that we have a decent normalization, constrain it to remain within 1/100th and 100 times
      #the current value (it improves A LOT the convergence speed, especially with MINUIT)
      for k,v in self.freeParameters.iteritems():
        value		      = v.value
        v.setBounds(value/100.0,value*100.0)
        v.setDelta(value/10.0)
      pass
    pass
    
    #Now perform the real fit
    
    #Get and store the parameters from the model manager
    freeParameters            = self.modelManager.getFreeParameters()
    self.freeParameters       = collections.OrderedDict()
    for k,v in freeParameters.iteritems():
      if(v.isNuisance()):
        continue
      else:
        self.freeParameters[k] = v
      pass
    pass
    
    minimizer                 = self.Minimizer(self.minusLogLikeProfile,self.freeParameters)
    xs,xserr,logLmin          = minimizer.minimize(minos,False)
    
    print("Minimum of -logLikelihood is: %s" %(logLmin))
    
    print("Contributions to the -logLikelihood at the minimum:")
    
    for dataset in self.dataSets:
      print("%-50s: %s" %(dataset.getName(),dataset.getLogLike()*(-1)))
    pass
    
    #Print and store results for future use
    print("\nValues for the parameters at the minimum are:")
    for i,(k,v) in enumerate(self.modelManager.getFreeParameters().iteritems()):
      if(v.isNuisance()):
        msg                   = "(nuisance)"
      else:
        msg                   = ''
      pass
      
      print("%-50s = %6.3g %s" %(k,v.value,msg))
      self.bestFitValues[k]   = v.value
      if(v.isNuisance()):
        self.approxErrors[k]  = 0
      else:
        self.approxErrors[k]    = xserr[i]
    pass
    self.logLmin              = logLmin
    
    return self.bestFitValues,logLmin    
  pass
  
  def _restoreBestFitValues(self):
    #Restore best fit values
    for k in self.freeParameters.keys():
      self.freeParameters[k].setValue(self.bestFitValues[k])
      self.modelManager[k].setValue(self.bestFitValues[k])
    pass  
  pass
  
  def getErrors(self,confidenceLevel=0.68268949213708585,**kwargs):
    '''
    Compute asymptotic errors using the Likelihood Ratio Test. Usage:
    
    computeErrors(0.68)
    
    will compute the 1-sigma error region, while:
    
    computeErrors(0.99)
    
    will compute the 99% c.l. error region, and so on. Alternatively, you
    can specify the number of sigmas corresponding to the desired c.l., as:
    
    computeErrors(sigma=1)
    
    to get the 68% c.l., or:
    
    computeErrors(sigma=2)
    
    to get the ~95% c.l.
    '''
    
    equivalentSigma           = None
    plotProfiles              = False
    for k,v in kwargs.iteritems():
      if(k.lower()=="sigma"):
        equivalentSigma       = float(v)
      elif(k.lower()=="profiles"):
        plotProfiles          = bool(v)
    pass
    
    if(confidenceLevel > 1.0 or confidenceLevel <= 0.0):
      raise RuntimeError("Confidence level must be 0 < cl < 1. Ex. use 0.683 for 1-sigma interval")
    
    #Get chisq critical value corresponding to this confidence level
    if(equivalentSigma==None):
      equivalentSigma         = scipy.stats.norm.isf((1-confidenceLevel)/2.0) 
    else:
      confidenceLevel         = 1-(scipy.stats.norm.sf(equivalentSigma)*2.0)
    pass
    
    criticalValue             = scipy.stats.chi2.isf([1-confidenceLevel],1)[0]
    
    print("Computing %.3f c.l. errors (chisq critical value: %.3f, equivalent sigmas: %.3f sigma)" %(confidenceLevel,criticalValue,equivalentSigma))
    
    #Now computing the errors
    if(len(self.bestFitValues.keys())==0):
      raise RuntimeError("You have to perform a fit before calling computeErrors!")
    
    #Find confidence intervals for all parameters, except nuisance ones
    paramNames                = self.bestFitValues.keys()
    paramList                 = []
    for par in paramNames:
      if(self.modelManager[par].isNuisance()):
        continue
      else:
        paramList.append(par)
      pass
    pass
    
    confInterval              = collections.OrderedDict()
    
    for i,parname in enumerate(paramList):
      
      sys.stdout.write("Computing error for parameter %s...\n" %(parname))
      
      #Get the list of free parameters
      self.freeParameters     = self.modelManager.getFreeParameters()
      
      self._restoreBestFitValues()
      
      #Remove the current parameter from the list of free parameters,
      #so that it won't be varied
      self.freeParameters.pop(parname)
      
      #Build the profile logLike for this parameter
      
      def thisProfileLikeRenorm(newValue):
        
        self._restoreBestFitValues()
        
        #Set the current parameter to its current value
        
        #newValue              = newValue[0]
        self.modelManager[parname].setValue(newValue)
        
        #Fit all other parameters
        
        minimizer             = self.Minimizer(self.minusLogLikeProfile,self.freeParameters)
        _,_,proflogL          = minimizer.minimize(False,False)
        
        #Subtract the minimum and the critical value/2, so that when this is 0 the true profileLogLike is 
        #logL+critical value/2.0
        #(the factor /2.0 comes from the LRT, which has 2*deltaLoglike as statistic)
        
        return proflogL-self.logLmin-criticalValue/2.0
      pass
      
      #Find the values of the parameter for which the profile logLike is
      # equal to the minimum - critical value/2.0, i.e., when thisProfileLikeRenorm is 0
      
      #We will use the approximate error (the sqrt of the diagonal of the covariance matrix)
      #as starting point for the search. Since it represents the approximate 1 sigma error,
      #we have to multiply it by the appropriate number of sigmas
      
      bounds                  = []
      for kind in ['lower','upper']:
        if(kind=='lower'):
          for i in [1.0,1.1,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]:
            approxSolution        = self.bestFitValues[parname]-i*equivalentSigma*abs(self.approxErrors[parname])
            if(thisProfileLikeRenorm(approxSolution) > 0):
              break
        else:
          for i in [1.0,1.1,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]:
            approxSolution        = self.bestFitValues[parname]+i*equivalentSigma*abs(self.approxErrors[parname])
            if(thisProfileLikeRenorm(approxSolution) > 0):
              break
        pass
        
        if(approxSolution < self.modelManager[parname].minValue):
          approxSolution          = self.modelManager[parname].minValue*1.1
        pass
        
        if(approxSolution > self.modelManager[parname].maxValue):
          approxSolution          = self.modelManager[parname].maxValue*0.9
        pass
        
        tolerance                 = abs(self.bestFitValues[parname])/10000.0
        
        if(self.verbose):
          print("Approx solution for %s bound is %s, tolerance is %s" %(kind,approxSolution,tolerance))
        try:
          #This find the root of thisProfileLikeRenorm, i.e., the value of its argument for which
          #it is zero
          #results                 = scipy.optimize.root(thisProfileLikeRenorm,
          #                                              approxSolution,
          #                                              method='lm')
          results                 = scipy.optimize.brentq(thisProfileLikeRenorm,approxSolution,self.bestFitValues[parname],rtol=1e-3)
          
        except:
          print("Error search for %s bound for parameter %s failed. Parameter is probably unconstrained." %(kind,parname))
          raise
        else:
          
          #if(results['success']==False):
          #  print RuntimeError("Could not find a solution for the %s bound confidence for parameter %s" %(kind,parname))
          #  raise
          #bounds.append(results['x'][0])
          bounds.append(results) 
      pass
                  
      confInterval[parname] = [min(bounds),max(bounds)]
    pass
    
    self.freeParameters     = self.modelManager.getFreeParameters()
    
    print("\nBest fit values and their errors are:")
    for parname in confInterval.keys():
      value                   = self.bestFitValues[parname]
      error1,error2           = confInterval[parname]
      print("%-20s = %6.3g [%6.4g,%6.4g]" %(parname,value,error1-value,error2-value))
    pass
    
    if(plotProfiles):
      #Plot the profile likelihoods for each parameter
      npar                    = len(confInterval.keys())
      nrows                   = npar/2
      ncols                   = 2
      if(nrows*ncols < npar):
        nrow                 += 1
      pass
      
      fig,subs                = plt.subplots(nrows,ncols)
      
      for i,sub,(parname,interval) in zip(range(npar),subs.flatten(),confInterval.iteritems()):
        #Remove this parameter from the freeParameters list
        #Get the list of free parameters
        self.freeParameters     = self.modelManager.getFreeParameters()
      
        self._restoreBestFitValues()
        
        #Remove the current parameter from the list of free parameters,
        #so that it won't be varied
        self.freeParameters.pop(parname)
        
        val                   = self.bestFitValues[parname]
        errorM                = interval[0]-val
        errorP                = interval[1]-val   
        grid                  = numpy.linspace(val+1.1*errorM,val+1.1*errorP,10)
        grid                  = numpy.append(grid,val)
        grid.sort()
        logLonGrid            = []
        for g in grid:
          self._restoreBestFitValues()
          logLonGrid.append(2*(thisProfileLikeRenorm(g)+criticalValue/2.0))
        pass
        sub.plot(grid,logLonGrid)
        sub.set_xlabel("%s" %(parname))
        sub.set_ylabel(r"2 ($L_{prof}$-L$_{0}$)")
        sub.axhline(criticalValue,linestyle='--')
        #Reduce the number of x ticks
        sub.locator_params(nbins=5,axis='x')
      pass
      
      plt.tight_layout()
      
    pass
    
    return confInterval
  pass
  
pass
