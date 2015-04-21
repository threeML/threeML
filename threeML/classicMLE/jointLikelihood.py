from threeML.minimizer import minimization
import collections

import numpy
import scipy.optimize
import scipy.stats
import sys
import matplotlib.pyplot as plt

from astropy.table import Table

from IPython.display import display

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
          
          globalLogLike         += dataset.innerFit()
          
          #dataset.getLogLike()      
      
      if("%s" % globalLogLike=='nan'):
        print("Warning: these parameters returned a logLike = Nan: %s" %(trialValues))
        return 1e6
      
      if(self.verbose):
        print("Trying with parameters %s, resulting in logL = %s" %(trialValues,globalLogLike))
      
      return globalLogLike*(-1)
  
  
  def setMinimizer(self,minimizer):
  
    if(minimizer.upper()=="IMINUIT"):
    
      self.Minimizer          = minimization.iMinuitMinimizer
    
    elif(minimizer.upper()=="MINUIT"):
    
      self.Minimizer          = minimization.MinuitMinimizer
    
    elif(minimizer.upper()=="SCIPY"):
    
      self.Minimizer          = minimization.ScipyMinimizer
    
    elif(minimizer.upper()=="BOBYQA"):
    
      self.Minimizer          = minimization.BOBYQAMinimizer
    
    else:
    
      raise ValueError("Do not know minimizer %s" %(minimizer))
  
  def fit(self):
        
    #Isolate the free parameters
    #NB: nuisance parameters are NOT in this dictionary
    
    self.freeParameters       = self.likelihoodModel.getFreeParameters()
    
    #Instance the minimizer
    self.minimizer            = self.Minimizer(self.minusLogLikeProfile,
                                               self.freeParameters)
    
    #Perform the fit
    xs, logLmin          = self.minimizer.minimize()
    
    #Print results
    print("Best fit values:\n")
    self.minimizer.printFitResults()
    
    print("\nCorrelation matrix:\n")
    self.minimizer.printCorrelationMatrix()
    
    print("\nMinimum of -logLikelihood is: %s\n" %(logLmin))
    
    print("Contributions to the -logLikelihood at the minimum:\n")
    
    data                      = []
    nameLength                = 0
    
    for dataset in self.dataList.values():
      
      nameLength              = max(nameLength, len(dataset.getName()) + 1)
      data.append([dataset.getName(),dataset.getLogLike()*(-1)])
    
    table                     = Table( rows  = data,
                                       names = ["Detector","-LogL"],
                                       dtype = ('S%i' %nameLength, float))
    
    display(table)
        
    return xs,logLmin    
  
  def getErrors(self,fast=False):
    
    if(not hasattr(self,'minimizer')):
      raise RuntimeError("You have to run the .fit method before calling errors.")
    
    return self.minimizer.getErrors(fast)
  
  def getLikelihoodProfiles(self):
    
    if(not hasattr(self,'minimizer')):
      raise RuntimeError("You have to run the .fit method before calling errors.")
    
    return self.minimizer.getLikelihoodProfiles()
  
  def getContours(self,param1,param2):
    
    return self.minimizer.getContours(param1,param2)
  
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

