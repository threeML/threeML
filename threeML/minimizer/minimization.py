import numpy
import itertools
import iminuit
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
  def __init__(self,function,parameters,ftol=1e-3,verbosity=0):
    
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
    #self.minuit.strategy      = 2 #More accurate
        
  def minimize(self,minos=False,printout=True,verbosity=0):
    
    self.minuit.migrad()
    
    return self.minuit.values, self.minuit.errors, self.minuit.fval
  
  def getErrors(self,fast): 
    
    #This is to get the covariance matrix right
    self.minuit.hesse()
    
    if(fast):
      #Do not execute MINOS
      return self.minuit.errors
    
    #Execute MINOS profiles
    parameters                     = self.minuit.list_of_vary_param()
    
    errors                         = {}
    figures                        = []
    
    for par in parameters:
      
      figures.append(plt.figure())
      
      errors[par]                  = self.minuit.draw_mnprofile(par, bound = 1.5, text=False)
    
    return errors, figures
  
  def getContours(self, param1, param2):
    
    #This is to get the covariance matrix right
    self.minuit.hesse()
    
    self.minuit.draw_mncontour(param1, param2, bins=100, nsigma=2, numpoints=20, sigma_res=4)
