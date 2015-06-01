import numpy
import itertools
import iminuit
import warnings
import matplotlib.pyplot as plt


class Minimizer(object):
  def __init__(self,function,parameters,ftol=1e-3,verbosity=1):
    
    self.function             = function
    self.parameters           = parameters
    self.Npar                 = len(self.parameters.keys())
    self.ftol                 = ftol
    self.verbosity            = verbosity
  
  def minimize(self):
    raise NotImplemented("This is the method of the base class. Must be implemented by the actual minimizer")

class iMinuitMinimizer(Minimizer):
  def __init__(self,function,parameters,ftol=1000,verbosity=0):
    
    super(iMinuitMinimizer, self).__init__(function,parameters,ftol,verbosity)
    
    #Prepare the parameter dictionary for iminuit
    pars                      = {}
    varnames                  = []
    
    for k,par in parameters.iteritems():
      
      curName                 = "%s of %s" %(k[1],k[0])
      
      varnames.append(curName)
      
      #Initial value
      pars['%s' % curName]       = par.value
      
      #Initial delta
      pars['error_%s' % curName] = par.delta
      
      #Limits
      pars['limit_%s' % curName] = [par.minValue,par.maxValue]
      
      #This is useless, since all parameters here are free,
      #but do it anyway for clarity
      pars['fix_%s' % curName]   = False
    
    
    #Since the number and names of the parameters cannot be extracted
    #from the signature of the likelihood function in input,
    #we use the forced_parameters facility which allows for their
    #explicit specification (see iMinuit manual for details)
    
    pars['forced_parameters'] = varnames 
    
    #This is to tell iMinuit that we are dealing with likelihoods,
    #not chi square
    pars['errordef']          = 0.5
        
    pars['print_level']       = verbosity
    
    #Finally we can instance the Minuit class
    self.minuit               = iminuit.Minuit(self.function, **pars)
    
    self.minuit.tol           = ftol
    self.minuit.strategy      = 1 #More accurate
        
  def minimize(self):
    
    #Repeat Migrad up to 10 times, until it converges
    for i in range(10):
      
      self.minuit.migrad()
      
      if(self.minuit.migrad_ok()):
        
        #Converged
        break
      
      else:
        
        #Try again
        continue
          
    return self.minuit.values, self.minuit.fval
  
  def printFitResults(self):
    
    self.minuit.print_param()
  
  def printCorrelationMatrix(self):
    
    self.minuit.print_matrix()
  
  def getErrors(self,fast): 
    
    #Set the print_level to 0 to avoid printing too many tables
    self.minuit.print_level = 0
    
    if(fast):
      
      #Do not execute MINOS
            
      #This is to get the covariance matrix right
      self.minuit.hesse()
      
      self.minuit.print_param()
      
      return self.minuit.errors
    
    else:
      
      errors = self.minuit.minos()
      
      self.minuit.print_param()
      
      return errors
  
  def getLikelihoodProfiles(self):
    #Execute MINOS profiles
    parameters                     = self.minuit.list_of_vary_param()
    
    errors                         = {}
    figures                        = []
    
    for par in parameters:
      
      figures.append(plt.figure())
      
      #Suppress UserWarnings, which can be too many due to migrad failing in region
      #of the parameter space too far from the best fit
      with warnings.catch_warnings() as w:
        
        warnings.filterwarnings("ignore",category=UserWarning)
        
        errors[par]                  = self.minuit.draw_mnprofile(par, bound = 1.2, text=False)
    
    return errors, figures
  
  def getContours(self, param1, param2):
    
    #This is to get the covariance matrix right
    self.minuit.hesse()
    
    self.minuit.draw_mncontour(param1, param2, bins=100, nsigma=2, numpoints=20, sigma_res=4)
