from threeML.minimizer import minimization
from threeML.exceptions import CustomExceptions
from threeML.io.Table import Table
from threeML.utils.cartesian import cartesian
from threeML.parallel.ParallelClient import ParallelClient
from threeML.io.ProgressBar import ProgressBar

import collections
import warnings
import copy

import numpy
import scipy.optimize
import scipy.stats
import sys, time
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

from IPython.display import display

import numpy as np

class JointLikelihood(object):
  
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
    
    #This is to keep track of the number of calls to the likelihood
    #function
    self.ncalls              = 0
    
    self.setMinimizer('iMinuit')
    
  pass
  
  def preFit(self):
    
    #This is a simple scan through the normalization parameters to guess
    #a good starting point for them (if we start too far from the data,
    #minuit and the other minimizers will have trouble)
    
    #Get the list of free parameters
    freeParams               = self.likelihoodModel.getFreeParameters()
    
    #Now isolate the normalizations, and use them as free parameters in the loglikelihood
    
    self.freeParameters      = collections.OrderedDict()
    
    for (k,v) in freeParams.iteritems():
      
      if v.isNormalization():
        
        self.freeParameters[k] = v
        
    #Prepare the grid of values to scan
    
    grids                    = []
    
    for norm in self.freeParameters.values():
            
      grids.append(numpy.logspace( numpy.log10( norm.minValue ),
                                   numpy.log10( norm.maxValue ), 
                                   50 ))
    
    if len(grids) == 0:
        
        #No norm. Maybe they are fixed ?
        
        return
    
    #Compute the global likelihood at each point in the grid
    globalGrid               = cartesian(grids)
    
    logLikes                 = map(self.minusLogLikeProfile, globalGrid)
    
    idx                      = numpy.argmin(logLikes)
    #print("Minimum is %s with %s" %(logLikes[idx],globalGrid[idx]))
    
    for i,norm in enumerate(self.freeParameters.values()):
      
      norm.setValue(globalGrid[idx][i])
      norm.setDelta(norm.value / 40)
    
  def minusLogLikeProfile(self, *trialValues):
      
      #Keep track of the number of calls
      self.ncalls                += 1
      
      #Assign the new values to the parameters
      
      for i,parname in enumerate(self.freeParameters.keys()):
        
        self.likelihoodModel.parameters[parname[0]][parname[1]].setValue(trialValues[i])
      
            
      #Now profile out nuisance parameters and compute the new value 
      #for the likelihood
      
      globalLogLike              = 0
      
      for dataset in self.dataList.values():
          
          try:
          
            globalLogLike         += dataset.innerFit()
          
          except CustomExceptions.ModelAssertionViolation:
            
            #This is a zone of the parameter space which is not allowed. Return
            #a big number for the likelihood so that the fit engine will avoid it
            warnings.warn("Fitting engine in forbidden space: %s" %(trialValues,))
            return 1e6
          
          except:
            
            #Do not intercept other errors
            raise
          
          #dataset.getLogLike()      
      
      if("%s" % globalLogLike=='nan'):
        warnings.warn("These parameters returned a logLike = Nan: %s" %(trialValues,))
        return 1e6
      
      if(self.verbose):
        print("Trying with parameters %s, resulting in logL = %s" %(trialValues,globalLogLike))
      
      return globalLogLike*(-1)
  
  
  def _setupMinimizer(self,minimizer):
  
    if(minimizer.upper()=="IMINUIT"):
    
      return minimization.iMinuitMinimizer
    
    elif(minimizer.upper()=="MINUIT"):
    
      return minimization.MinuitMinimizer
    
    elif(minimizer.upper()=="SCIPY"):
    
      return minimization.ScipyMinimizer
    
    elif(minimizer.upper()=="BOBYQA"):
    
      return minimization.BOBYQAMinimizer
    
    else:
    
      raise ValueError("Do not know minimizer %s" %(minimizer))
  
  def setMinimizer(self, minimizer):
    
    self.Minimizer = self._setupMinimizer(minimizer)
  
  def _getTableOfParameters(self,parameters):
    
    data = []
    nameLength = 0
    
    for k,v in parameters.iteritems():
     
     curName                 = "%s_of_%s" %(k[1],k[0])
               
     data.append([curName,"%s" % v.value,v.unit])
     
     if(len(v.name) > nameLength):
       nameLength = len(curName)
   
    pass
    
    table                     = Table(rows = data,
                                      names = ["Name","Value","Unit"],
                                      dtype=('S%i' %nameLength, str, 'S15'))
    
    return table

  
  def fit(self):
    
    #Pre-fit: will fit the normalizations so that they are not too far
    #from the data (which would make the fitting below fail)
    self.preFit()
    
    #Isolate the free parameters
    #NB: nuisance parameters are NOT in this dictionary
    
    self.freeParameters       = self.likelihoodModel.getFreeParameters()
    
    #Now check and fix if needed all the deltas of the parameters
    #to 10% of their value (otherwise the fit will be super-slow)
    for k,v in self.freeParameters.iteritems():
      
      if (abs(v.delta) < abs(v.value) * 0.1):
                
        v.setDelta(abs(v.value) * 0.1)
    
    #Instance the minimizer
    self.minimizer            = self.Minimizer(self.minusLogLikeProfile,
                                               self.freeParameters)
    
    #Perform the fit
    xs, logLmin          = self.minimizer.minimize()
    
    #Print results
    print("Best fit values:\n")
    
    self.minimizer.printFitResults()
    
    print("Nuisance parameters:\n")
    
    nuisanceParam = collections.OrderedDict()
    
    for dataset in self.dataList.values():
      
      for pName in dataset.getNuisanceParameters():
        
        nuisanceParam[ ( dataset.getName(), pName )] = dataset.nuisanceParameters[pName]
    
    table = self._getTableOfParameters(nuisanceParam)
    display(table)
    
    print("\nCorrelation matrix:\n")
    
    self.minimizer.printCorrelationMatrix()
    
    print("\nMinimum of -logLikelihood is: %s\n" %(logLmin))
    
    print("Contributions to the -logLikelihood at the minimum:\n")
    
    mLogLikes                 = collections.OrderedDict()
    mLogLikes['total']        = logLmin
    
    self.currentMinimum       = float(logLmin)
    
    data                      = []
    nameLength                = 0
    
    for dataset in self.dataList.values():
      
      ml                      = dataset.getLogLike()*(-1)
      
      mLogLikes[dataset.getName()] = ml
      
      nameLength              = max(nameLength, len(dataset.getName()) + 1)
      data.append( [ dataset.getName(), ml ] )
    
    table                     = Table( rows  = data,
                                       names = ["Detector","-LogL"],
                                       dtype = ('S%i' %nameLength, float))
          
    table['-LogL'].format = '2.2f'
    
    display(table)
    
    #Prepare the dictionary with the results
    results                   = collections.OrderedDict()
    
    results['parameters']     = xs
    results['minusLogLike']   = mLogLikes
    
    return results
  
  def getErrors(self,fast=False):
    
    if(not hasattr(self,'currentMinimum')):
      raise RuntimeError("You have to run the .fit method before calling errors.")
        
    return self.minimizer.getErrors(fast)
  
  def explore(self):
      
      errors = self.getErrors(fast=True)
      
      import emcee
      import emcee.utils
      
      #Dimensionality of the problem
      ndim = len(self.minimizer.parameters.values())
      
      #emcee wants the +logLike, and has a different
      #calling sequence. This is just a wrapper around
      #the likelihood
      
      def funwrap(trialList):
        
        return self.minimizer.function(*trialList)*(-1)
      
      #How many walkers?
      nwalkers = ndim * 8
      
      #Prepare the start values for the walkers
      
      #Best fit values
      p0_ = numpy.asarray(map(lambda x:x.value, self.minimizer.parameters.values()))
      
      #Randomize using the errors on the parameters from the fit
      
      deltas                  = []
    
      for k,par in self.minimizer.parameters.iteritems():
      
        curName                 = "%s_of_%s" %(k[1],k[0])
              
        deltas.append(errors[curName])
      
      p0 = emcee.utils.sample_ball(p0_, deltas, nwalkers)
      
      #Ensure that at least one of the p0 is the best fit
      p0[0] = p0_
      
      #Ensure that the starting points are good
      logLike = map(funwrap, p0)
      
      idx = numpy.isfinite(logLike)
      
      if( numpy.sum(~idx) > 0 ):
                
        p0 = p0[idx]
        nwalkers = p0.shape[0]
        
        #nwalkers must be even
        if( nwalkers % 2 != 0):
          p0 = p0[1:]
          nwalkers = p0.shape[0]
        
        #There is either inf or Nan in logLike,
        #one or more of the starting points are invalid
        warnings.warn("One or more starting points for the MCMC are invalid. Removing them. Walkers are now %s" %(nwalkers))
      pass
      
      sampler = emcee.EnsembleSampler(nwalkers, len(self.minimizer.parameters.values()), funwrap)
      
      #Burn-in
      pos, prob, state = sampler.run_mcmc(p0, 100)
      
      sampler.reset()
      
      #Real sampling
      sampler.run_mcmc(pos, 1000)
      
      return sampler
  
  def getLikelihoodProfiles(self):
    
    if(not hasattr(self,'minimizer')):
      raise RuntimeError("You have to run the .fit method before calling errors.")
    
    return self.minimizer.getLikelihoodProfiles()
  
  def getContours(self, src1, param1, p1min, p1max, p1steps,
                        src2=None, param2=None, p2min=None, p2max=None, p2steps=None,
                        **kwargs):
    
    if(not hasattr(self,'currentMinimum')):
      raise RuntimeError("You have to run the .fit method before calling getContours.")
    
    
    #Default values
    threads = 1
    debug = False
    
    #Check the current keywords
    for k,v in kwargs.iteritems():
        
        if k=="threads":
            
            threads = int( v )
        
        elif k=="debug":
            
            debug = bool( v )
    
    
    #Check that parameters are existent and free
    for s,p in zip( [ src1, src2 ], [ param1, param2 ] ):
      
      if s is None:
          
          #Stepping through one parameter only
          continue
      
      if ( (s, p) not in self.freeParameters.keys() ):
           
           raise ValueError("Parameter %s of source %s is not a free parameter of current model" %(p,s))
    
    if(threads <= 1):
      #Create a new minimizer to avoid messing up with the best
      #fit
      
      #Copy of the parameters
      backup_freeParameters = copy.deepcopy(self.freeParameters)
      
      minimizer = self.Minimizer(self.minusLogLikeProfile,
                                 self.freeParameters)
      
      a, b, cc = minimizer.contours(src1, param1, p1min, p1max, p1steps,
                                    src2, param2, p2min, p2max, p2steps,
                                    True, **kwargs)
      
      #Restore the original
      self.freeParameters = backup_freeParameters
                                             
    #if( 2==2 ):
    else:
      
      if( threads > p1steps):
        
        threads = int(p1steps)
        
        warnings.warn("The number of threads is larger than the number of steps. Reducing it to %s." %(threads))
                  
      #Check if the number of steps is divisible by the number
      #of threads, otherwise issue a warning and make it so
      
      if( float(p1steps) % threads != 0 ):
        
        p1steps = p1steps // threads * threads
        
        warnings.warn("The number of steps is not a multiple of the number of threads. Reducing steps to %s" %(p1steps))
      
      #Now this is guaranteed to be a integer number
      p1_split_steps = p1steps // int(threads)
      
      #Prepare arrays for results
      
      if src2 is None:
          
          #One array
          pcc = pcc = numpy.zeros( ( p1steps ) )
          
          pa = numpy.linspace( p1min, p1max, p1steps )
          pb = None
      
      else:
      
          pcc = numpy.zeros( ( p1steps, p2steps ) )
      
          #Prepare the two axes of the parameter space
          pa = numpy.linspace( p1min, p1max, p1steps )
          pb = numpy.linspace( p2min, p2max, p2steps )
      
      #Define the parallel worker which will go through the computation
      
      #NOTE: I only divide
      #on the first parameter axis so that the different
      #threads are more or less well mixed for points close and
      #far from the best fit
      
      def worker(i):
        
        #Re-create the minimizer
        
        #backup_freeParameters = copy.deepcopy(self.freeParameters)
        
        minimizer = self.Minimizer(self.minusLogLikeProfile,
                                   self.freeParameters)
        
        this_p1min = pa[i * p1_split_steps]
        this_p1max = pa[(i+1) * p1_split_steps - 1]
        
        if debug:
            
            print("From %s to %s" %(this_p1min, this_p1max))
        
        aa, bb, ccc = minimizer.contours(src1, param1, this_p1min, this_p1max, p1_split_steps,
                                         src2, param2, p2min, p2max, p2steps,
                                         False, **kwargs)
        
        #self.freeParameters = backup_freeParameters
        
        return ccc
      
      try:
        
        client = ParallelClient(**kwargs)
      
      except:
        
        sys.stderr.write("\nCannot connect to IPython cluster. Is the cluster running ?\n\n\n")
        
        raise RuntimeError("Cannot connect to IPython cluster.")
        
      lview = client.load_balanced_view()
      #lview.block = True
      amr = lview.map_async(worker, range(threads))
      
      #Execute and print progress
      
      prog = ProgressBar(threads)
      
      while not amr.ready():
        
        #Avoid checking too often
        time.sleep(1)
        
        if(debug):
          stdouts = amr.stdout
          
          # clear_output doesn't do much in terminal environments
          for stdout, stderr in zip(amr.stdout, amr.stderr):
            if stdout:
                print "%s" % (stdout[-1000:])
            if stderr:
                print "%s" % (stderr[-1000:])
          sys.stdout.flush()        
        
        prog.animate( amr.progress )
      
      #Force to display 100% at the end
      prog.animate( threads - 1 )
      
      #Now get results and print some diagnostic
      print("\n")
      print("Serial time: %1.f (speed-up: %.1f)" %(amr.serial_time, float(amr.serial_time) / amr.wall_time))
      res = amr.get()
      
      for i in range(threads):
        
        if src2 is None:
        
            pcc[ i * p1_split_steps : (i+1) * p1_split_steps ] = res[i][:,0]
        
        else:
        
            pcc[ i * p1_split_steps : (i+1) * p1_split_steps, : ] = res[i]
      
      #Keep them separated up to now for debugging purposes
      cc = pcc
      a = pa
      b = pb
      
    pass
    
    if src2 is not None:
    
        fig = self._plotContours("%s of %s" %(param1, src1), a, "%s of %s" % (param2, src2), b, cc)
    
    else:
        
        fig = self._plotProfile( "%s of %s" %(param1, src1), a, cc )
    
    #Check if we found a better minimum
    if( self.currentMinimum - cc.min() > 0.1 ):
      
      if src2 is not None:
      
          idx = cc.argmin()
          
          aidx, bidx = numpy.unravel_index(idx, cc.shape)
          
          print("Found a better minimum: %s with %s = %s and %s = %s" 
                 %(cc.min(),param1,a[aidx],param2,b[bidx]))
      
      else:
          
          idx = cc.argmin()
          
          print("Found a better minimum: %s with %s = %s" 
                 %(cc.min(),param1,a[idx]))
    
    pass
    
    return a, b, cc, fig
  
  def _plotProfile(self, name1, a, cc):
      
      #plot 1,2 and 3 sigma horizontal lines
      sigmas = [1,2,3]
      
      #Compute the corresponding probability. We do not
      #pre-compute them because we will introduce at
      #some point the possibility to personalize the
      #levels
      probs = []
      
      for s in sigmas:
          
          #One-sided probability
          #It is one-sided because we consider one side at the time
          #when computing the error
          
          probs.append( 1 - (scipy.stats.norm.sf(s) * 2) )
      
      #Compute the corresponding delta chisq. (chisq has 1 d.o.f.)
      
      deltachi2 = scipy.stats.chi2.ppf(probs, 1)
      
      fig = plt.figure()
      sub = fig.add_subplot(111)
     
      sub.plot( a, cc, lw=2)
      
      #Decide colors
      colors = ['blue','cyan','red']
      
      for s,d,c in zip( sigmas, deltachi2, colors ):
          
          sub.axhline( self.currentMinimum + d , linestyle='--' , 
                       color=c, label=r'%s $\sigma$' %( s ), lw = 2 )
      
      plt.legend( loc=0, frameon=True )
      
      sub.set_xlabel( name1 )
      sub.set_ylabel( "-log( likelihood )" )
      
      return fig
      
  
  def _plotContours(self, name1, a, name2, b, cc):
      
      #plot 1,2 and 3 sigma contours
      sigmas = [1,2,3]
      
      #Compute the corresponding probability. We do not
      #pre-compute them because we will introduce at
      #some point the possibility to personalize the
      #levels
      probs = []
      
      for s in sigmas:
          
          #One-sided probability
          #It is one-sided because we consider one side at the time
          #when computing the error
          
          probs.append( 1 - (scipy.stats.norm.sf(s) * 2) )
      
      #Compute the corresponding delta chisq. (chisq has 2 d.o.f.)
      deltachi2 = scipy.stats.chi2.ppf(probs, 2)
      
      #Boundaries for the colormap
      bounds = [self.currentMinimum]
      bounds.extend(self.currentMinimum + deltachi2)
      bounds.append(cc.max())
      
      #Define the color palette
      palette = cm.Pastel1
      palette.set_over('white')
      palette.set_under('white')
      palette.set_bad('white')
      
      fig = plt.figure()
      sub = fig.add_subplot(111)
      
      #Show the contours with square axis
      im = sub.imshow(cc,
                 cmap=palette,
                 extent=[b.min(),b.max(),a.min(),a.max()],
                 aspect=(b.max()-b.min())/(a.max()-a.min()),
                 origin='lower',
                 norm=BoundaryNorm(bounds,256),
                 interpolation='bicubic',
                 vmax=(self.currentMinimum+deltachi2).max())
      
      #Plot the color bar with the sigmas
      cb = fig.colorbar(im, boundaries=bounds[:-1])
      lbounds = [0]
      lbounds.extend(bounds[:-1])
      cb.set_ticks(lbounds)
      ll = ['']
      ll.extend(map(lambda x:r'%i $\sigma$' %x, sigmas))
      cb.set_ticklabels(ll)
      
      #Align the labels to the end of the color level
      for t in cb.ax.get_yticklabels():
          t.set_verticalalignment('baseline')   
      
      #Draw the line contours
      sub.contour(b,a, cc, self.currentMinimum + deltachi2)
      
      sub.set_xlabel(name2)
      sub.set_ylabel(name1)
      
      return fig
      
#  def _restoreBestFitValues(self):
#    #Restore best fit values
#    for k in self.freeParameters.keys():
#      self.freeParameters[k].setValue(self.bestFitValues[k])
#      self.modelManager[k].setValue(self.bestFitValues[k])
#    pass  
#  pass
  
#  def getErrors(self,confidenceLevel=0.68268949213708585,**kwargs):
#    '''
#    Compute asymptotic errors using the Likelihood Ratio Test. Usage:
#    
#    computeErrors(0.68)
#    
#    will compute the 1-sigma error region, while:
#    
#    computeErrors(0.99)
#    
#    will compute the 99% c.l. error region, and so on. Alternatively, you
#    can specify the number of sigmas corresponding to the desired c.l., as:
#    
#    computeErrors(sigma=1)
#    
#    to get the 68% c.l., or:
#    
#    computeErrors(sigma=2)
#    
#    to get the ~95% c.l.
#    '''
#    
#    equivalentSigma           = None
#    plotProfiles              = False
#    for k,v in kwargs.iteritems():
#      if(k.lower()=="sigma"):
#        equivalentSigma       = float(v)
#      elif(k.lower()=="profiles"):
#        plotProfiles          = bool(v)
#    pass
#    
#    if(confidenceLevel > 1.0 or confidenceLevel <= 0.0):
#      raise RuntimeError("Confidence level must be 0 < cl < 1. Ex. use 0.683 for 1-sigma interval")
#    
#    #Get chisq critical value corresponding to this confidence level
#    if(equivalentSigma==None):
#      equivalentSigma         = scipy.stats.norm.isf((1-confidenceLevel)/2.0) 
#    else:
#      confidenceLevel         = 1-(scipy.stats.norm.sf(equivalentSigma)*2.0)
#    pass
#    
#    criticalValue             = scipy.stats.chi2.isf([1-confidenceLevel],1)[0]
#    
#    print("Computing %.3f c.l. errors (chisq critical value: %.3f, equivalent sigmas: %.3f sigma)" %(confidenceLevel,criticalValue,equivalentSigma))
#    
#    #Now computing the errors
#    if(len(self.bestFitValues.keys())==0):
#      raise RuntimeError("You have to perform a fit before calling computeErrors!")
#    
#    #Find confidence intervals for all parameters, except nuisance ones
#    paramNames                = self.bestFitValues.keys()
#    paramList                 = []
#    for par in paramNames:
#      if(self.modelManager[par].isNuisance()):
#        continue
#      else:
#        paramList.append(par)
#      pass
#    pass
#    
#    confInterval              = collections.OrderedDict()
#    
#    for i,parname in enumerate(paramList):
#      
#      sys.stdout.write("Computing error for parameter %s...\n" %(parname))
#      
#      #Get the list of free parameters
#      self.freeParameters     = self.modelManager.getFreeParameters()
#      
#      self._restoreBestFitValues()
#      
#      #Remove the current parameter from the list of free parameters,
#      #so that it won't be varied
#      self.freeParameters.pop(parname)
#      
#      #Build the profile logLike for this parameter
#      
#      def thisProfileLikeRenorm(newValue):
#        
#        self._restoreBestFitValues()
#        
#        #Set the current parameter to its current value
#        
#        #newValue              = newValue[0]
#        self.modelManager[parname].setValue(newValue)
#        
#        #Fit all other parameters
#        
#        minimizer             = self.Minimizer(self.minusLogLikeProfile,self.freeParameters)
#        _,_,proflogL          = minimizer.minimize(False,False)
#        
#        #Subtract the minimum and the critical value/2, so that when this is 0 the true profileLogLike is 
#        #logL+critical value/2.0
#        #(the factor /2.0 comes from the LRT, which has 2*deltaLoglike as statistic)
#        
#        return proflogL-self.logLmin-criticalValue/2.0
#      pass
#      
#      #Find the values of the parameter for which the profile logLike is
#      # equal to the minimum - critical value/2.0, i.e., when thisProfileLikeRenorm is 0
#      
#      #We will use the approximate error (the sqrt of the diagonal of the covariance matrix)
#      #as starting point for the search. Since it represents the approximate 1 sigma error,
#      #we have to multiply it by the appropriate number of sigmas
#      
#      bounds                  = []
#      for kind in ['lower','upper']:
#        if(kind=='lower'):
#          for i in [1.0,1.1,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]:
#            approxSolution        = self.bestFitValues[parname]-i*equivalentSigma*abs(self.approxErrors[parname])
#            if(thisProfileLikeRenorm(approxSolution) > 0):
#              break
#        else:
#          for i in [1.0,1.1,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]:
#            approxSolution        = self.bestFitValues[parname]+i*equivalentSigma*abs(self.approxErrors[parname])
#            if(thisProfileLikeRenorm(approxSolution) > 0):
#              break
#        pass
#        
#        if(approxSolution < self.modelManager[parname].minValue):
#          approxSolution          = self.modelManager[parname].minValue*1.1
#        pass
#        
#        if(approxSolution > self.modelManager[parname].maxValue):
#          approxSolution          = self.modelManager[parname].maxValue*0.9
#        pass
#        
#        tolerance                 = abs(self.bestFitValues[parname])/10000.0
#        
#        if(self.verbose):
#          print("Approx solution for %s bound is %s, tolerance is %s" %(kind,approxSolution,tolerance))
#        try:
#          #This find the root of thisProfileLikeRenorm, i.e., the value of its argument for which
#          #it is zero
#          #results                 = scipy.optimize.root(thisProfileLikeRenorm,
#          #                                              approxSolution,
#          #                                              method='lm')
#          results                 = scipy.optimize.brentq(thisProfileLikeRenorm,approxSolution,self.bestFitValues[parname],rtol=1e-3)
#          
#        except:
#          print("Error search for %s bound for parameter %s failed. Parameter is probably unconstrained." %(kind,parname))
#          raise
#        else:
#          
#          #if(results['success']==False):
#          #  print RuntimeError("Could not find a solution for the %s bound confidence for parameter %s" %(kind,parname))
#          #  raise
#          #bounds.append(results['x'][0])
#          bounds.append(results) 
#      pass
#                  
#      confInterval[parname] = [min(bounds),max(bounds)]
#    pass
#    
#    self.freeParameters     = self.modelManager.getFreeParameters()
#    
#    print("\nBest fit values and their errors are:")
#    for parname in confInterval.keys():
#      value                   = self.bestFitValues[parname]
#      error1,error2           = confInterval[parname]
#      print("%-20s = %6.3g [%6.4g,%6.4g]" %(parname,value,error1-value,error2-value))
#    pass
#    
#    if(plotProfiles):
#      #Plot the profile likelihoods for each parameter
#      npar                    = len(confInterval.keys())
#      nrows                   = npar/2
#      ncols                   = 2
#      if(nrows*ncols < npar):
#        nrow                 += 1
#      pass
#      
#      fig,subs                = plt.subplots(nrows,ncols)
#      
#      for i,sub,(parname,interval) in zip(range(npar),subs.flatten(),confInterval.iteritems()):
#        #Remove this parameter from the freeParameters list
#        #Get the list of free parameters
#        self.freeParameters     = self.modelManager.getFreeParameters()
#      
#        self._restoreBestFitValues()
#        
#        #Remove the current parameter from the list of free parameters,
#        #so that it won't be varied
#        self.freeParameters.pop(parname)
#        
#        val                   = self.bestFitValues[parname]
#        errorM                = interval[0]-val
#        errorP                = interval[1]-val   
#        grid                  = numpy.linspace(val+1.1*errorM,val+1.1*errorP,10)
#        grid                  = numpy.append(grid,val)
#        grid.sort()
#        logLonGrid            = []
#        for g in grid:
#          self._restoreBestFitValues()
#          logLonGrid.append(2*(thisProfileLikeRenorm(g)+criticalValue/2.0))
#        pass
#        sub.plot(grid,logLonGrid)
#        sub.set_xlabel("%s" %(parname))
#        sub.set_ylabel(r"2 ($L_{prof}$-L$_{0}$)")
#        sub.axhline(criticalValue,linestyle='--')
#        #Reduce the number of x ticks
#        sub.locator_params(nbins=5,axis='x')
#      pass
#      
#      plt.tight_layout()
#      
#    pass
#    
#    return confInterval
#  pass

