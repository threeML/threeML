import numpy
import math

import warnings
import matplotlib.pyplot as plt
import collections

from threeML.io.Table import Table, NumericMatrix
from threeML.utils.cartesian import cartesian
from threeML.io.ProgressBar import ProgressBar

from iminuit import Minuit

from IPython.display import display

import uncertainties

import re

#Special constants
FIT_FAILED = -1e12

class Minimizer(object):
  def __init__(self,function,parameters,ftol=1e-3,verbosity=1):
    
    self.function             = function
    self.parameters           = parameters
    self.Npar                 = len(self.parameters.keys())
    self.ftol                 = ftol
    self.verbosity            = verbosity
  
  def minimize(self):
    raise NotImplemented("This is the method of the base class. Must be implemented by the actual minimizer")

#This is a function to add a method to a class
#We will need it in the MinuitMinimizer

def add_method(self, method, name=None):
    if name is None:
        name = method.func_name
    setattr(self.__class__, name, method)


class iMinuitMinimizer(Minimizer):
  def __init__(self,function,parameters,ftol=1e3,verbosity=0):
    
    super(iMinuitMinimizer, self).__init__(function,parameters,ftol,verbosity)
    
    #Prepare the parameter dictionary for iminuit
    pars                      = {}
    varnames                  = []
    
    for k,par in parameters.iteritems():
      
      curName                 = "%s_of_%s" %(k[1],k[0])
      
      varnames.append(curName)
      
      #Initial value
      pars['%s' % curName]       = par.value
      
      #Initial delta
      pars['error_%s' % curName] = par.delta
      
      #Limits
      pars['limit_%s' % curName] = ( par.minValue, par.maxValue )
      
      #This is useless, since all parameters here are free,
      #but do it anyway for clarity
      pars['fix_%s' % curName]   = False
    
    #This is to tell Minuit that we are dealing with likelihoods,
    #not chi square
    pars['errordef']          = 0.5
        
    pars['print_level']       = verbosity
    
    #We need to make a function with the parameters as explicit
    #variables in the calling sequence, so that Minuit will be able
    #to probe the parameter's names
    var_spelledout            = ",".join(varnames)
    
    #A dictionary to keep a way to convert from var. name to
    #variable position in the function calling sequence 
    #(will use this in contours)
    
    self.nameToPos = { k: i for i, k in enumerate( varnames ) }
    
    #Write and compile the code for such function
      
    code                      = 'def _f(self, %s):\n  return self.function(%s)' % (var_spelledout, var_spelledout)
    exec(code)
    
    #Add the function just created as a method of the class
    #so it will be able to use the 'self' pointer
    add_method(self, _f, "_f")    
        
    #Finally we can instance the Minuit class
    self.minuit               = Minuit(self._f, **pars)
        
    self.minuit.tol           = 100 #ftol
    
    try:
        
        self.minuit.up            = 0.5 #This is a likelihood
    
    except:
        
        #iMinuit uses errodef, not up
        
        self.minuit.errordef  = 0.5
    
    self.minuit.strategy      = 0 #More accurate
    
  def migradConverged(self):
    
    #In the MINUIT manual this is the condition for MIGRAD to have converged
    #0.002 * tolerance * UPERROR (which is 0.5 for likelihood)
    return ( self.minuit.edm <= 0.002 * self.minuit.tol * 0.5)
  
  
  def _runMigrad(self, trials=2):
    
    #Repeat Migrad up to 10 times, until it converges
    
    for i in range(trials):
      
      self.minuit.migrad()
      
      if self.migradConverged():
        
        #Converged
        break
      
      else:
        
        #Try again
        continue
  
  def minimize(self):
    
    self._runMigrad()
    
    if not self.migradConverged():
      
      print("\nMIGRAD did not converge in 10 trials.")
      
      return map(lambda x:0,self.minuit.values), 1e9
    
    else:
            
      #Make a ordered dict for the results
      bestFit                   = collections.OrderedDict()
      
      for k,par in self.parameters.iteritems():
        
        curName                 = "%s_of_%s" %(k[1],k[0])
        
        bestFit[curName]        = self.minuit.values[curName]
      
      #NOTE: hesse must be callsed AFTER the fit because it
      #will change the value of the parameters
      
      self.minuit.hesse()
      
      #Restore best fit
      for k,par in self.parameters.iteritems():
        
        curName                 = "%s_of_%s" %(k[1],k[0])
        
        par.setValue( bestFit[curName] )
      
      return bestFit, self.minuit.fval
  
  def printFitResults(self):
    
    #I do not use the print_param facility in iminuit because
    #it does not work well with console output, since it fails
    #to autoprobe that it is actually run in a console and uses
    #the HTML backend instead
    
    data = []
    nameLength = 0
    
    for k,v in self.parameters.iteritems():
     
     curName                 = "%s_of_%s" %(k[1],k[0])
     
     #Format the value and the error with sensible significant
     #numbers
     x                      = uncertainties.ufloat(v.value, self.minuit.errors[curName])
     rep                    = x.__str__().replace("+/-"," +/- ")
          
     data.append([curName,rep,v.unit])
     
     if(len(v.name) > nameLength):
       nameLength = len(curName)
   
    pass
    
    table                     = Table(rows = data,
                                      names = ["Name","Value","Unit"],
                                      dtype=('S%i' %nameLength, str, 'S15'))
    
    display(table)
    print("\nNOTE: errors on parameters are approximate. Use getErrors().\n")
      
  def printCorrelationMatrix(self):
        
    #Print a custom covariance matrix because iminuit does
    #not guess correctly the frontend when 3ML is used
    #from terminal
    
    cov                       = self.minuit.covariance
    
    if cov is None:
        
        raise RuntimeError("Cannot compute covariance. This usually means that there are unconstrained" + 
                           " parameters. Fix those or reduce their allowed range, or use a simpler model.")
    
    keys                      = self.parameters.keys()
    
    parNames                  = map(lambda k:"%s_of_%s" %(k[1],k[0]), keys ) 
    
    data = []
    nameLength = 0
    
    for key1, name1 in zip(keys, parNames):
       
      if( len(name1) > nameLength ):
        
        nameLength = len(name1)
     
      thisRow                 = []
      
      v1                      = self.parameters[key1]
      
      for key2, name2 in zip(keys, parNames):
       
        corr                  = cov[(name1, name2)] / ( math.sqrt(cov[(name1, name1)]) * math.sqrt(cov[(name2, name2)]) ) 
       
        thisRow.append(corr)
      
      pass
      
      data.append(thisRow)
      
    pass
    
    dtypes                    = []
    dtypes.extend( map(lambda x:float, parNames) )
    
    cols                      = []
    cols.extend(parNames)
    
    table                     = NumericMatrix(rows = data,
                                      names = cols,
                                      dtype= dtypes)
    
    for col in table.colnames:
      
      table[col].format = '2.2f'
    
    display(table)
      
  def getErrors(self,fast): 
    
    #Run again the fit because the user might have changed the parameter
    #configuration
    self._runMigrad()
    
    #Now set aside the current values for the parameters,
    #because minos will change them
    #Make a ordered dict for the results
    bestFit                   = collections.OrderedDict()
    
    for k,par in self.parameters.iteritems():
      
      curName                 = "%s_of_%s" %(k[1],k[0])
      
      bestFit[curName]        = self.minuit.values[curName]

    
    if not self.migradConverged():
      
      print("\nMIGRAD results are not valid. Cannot compute errors. Did you run the fit first ?")
      
      return map(lambda x:None, self.parameters.keys())
    
    try:
    
      self.minuit.minos()
    
    except:
      
      print("MINOS has failed. This usually means that the fit is very difficult, for example "
            "because of high correlation between parameters. Check the correlation matrix printed"
            "in the fit step, and check contour plots with getContours(). If you are using a "
            "user-defined model, you can also try to "
            "reformulate your model with less correlated parameters.")
      
      return None
          
    #Make a ordered dict for the results
    errors                   = collections.OrderedDict()
    
    for k,par in self.parameters.iteritems():
      
      curName                 = "%s_of_%s" %(k[1],k[0])
      
      errors[curName]         = (self.minuit.merrors[(curName, -1)], self.minuit.merrors[(curName, 1)])
      
      #Set the parameter back to the best fit value
      par.setValue( bestFit[curName] )
    
    
    #Print a table with the errors
    data = []
    nameLength = 0
    
    for k,v in self.parameters.iteritems():
    
      curName                 = "%s_of_%s" %(k[1],k[0])
      
      #Format the value and the error with sensible significant
      #numbers
      x                      = uncertainties.ufloat(v.value, abs(errors[curName][0]))
      
      num, uncm, ex          = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                        x.__str__().replace("+/-"," ")).groups()
      
      x                      = uncertainties.ufloat(v.value, abs(errors[curName][1]))
      _, uncp, _             = re.match('\(?(\-?[0-9]+\.?[0-9]+) ([0-9]+\.[0-9]+)\)?(e[\+|\-][0-9]+)?',
                                        x.__str__().replace("+/-"," ")).groups()
      
      if(ex==None):
        
        #Number without exponent
        prettystring           = "%s -%s +%s" %(num, uncm, uncp)
      
      else:
        
        prettystring           = "(%s -%s +%s)%s" %(num, uncm, uncp, ex)
        
      data.append([curName,prettystring,v.unit])
     
      if(len(v.name) > nameLength):
        nameLength = len(curName)
    pass
    
    table                     = Table(rows = data,
                                      names = ["Name","Value","Unit"],
                                      dtype=('S%i' %nameLength, str, 'S15'))
    
    display(table)

    
    return errors
  
  def contours(self, src1, param1, p1min, p1max, p1steps,
                    src2, param2, p2min, p2max, p2steps,
                    progress=True, **kwargs):
    
    #First create another minimizer 
    
    newargs = dict( self.minuit.fitarg )
    
    #Update the values for the parameters with the best fit one
    
    for key, value in self.minuit.values.iteritems():
        
        newargs[key] = value
    
    #Fix the parameters under scrutiny
    
    #values = {}
    
    for s,p in zip( [src1, src2], [param1,param2] ):
        
        if s is None:
            
            #Only one parameter to analyze
            
            continue
        
        key = "%s_of_%s" %(p,s)
        
        if key not in newargs.keys():
            
            raise ValueError("Parameter %s is not a free parameter for source %s." %(p,s))
        
        else:
            
            newargs[ 'fix_%s' % key ] = True
        
        #values[ key ] = float( self.minuit.values[key] )
    
    #This is a likelihood
    newargs['errordef'] = 0.5
    
    newargs['print_level'] = 0
        
    #Now create the new minimizer
    self.contour_minuit = Minuit( self._f, **newargs )
    
    #Check the keywords
    p1log = False
    p2log = False
    
    if 'log' in kwargs.keys():
        
        p1log = bool( kwargs['log'][0] )
        
        if param2 is not None:
            
            p2log = bool( kwargs['log'][1] )
    
    #Generate the steps
    
    if p1log:
    
        a = numpy.logspace( numpy.log10(p1min), numpy.log10(p1max), p1steps)
    
    else:
        
        
        a = numpy.linspace( p1min, p1max, p1steps)
    
    if param2 is not None:
    
        if p2log:
        
            b = numpy.logspace( numpy.log10(p2min), numpy.log10(p2max), p2steps)
        
        else:
        
            b = numpy.linspace( p2min, p2max, p2steps)
    
    else:
        
        #Only one parameter to step through
        #Put b as nan so that the worker can realize that it does not have
        #to step through it
        
        b = numpy.array([numpy.nan])
    
    #Generate the grid
    
    grid = cartesian([a,b])
    
    #Define the parallel worker
    
    def contourWorker(args):
            
      aa, bb = args
      
      name1 = "%s_of_%s" % ( param1, src1 )
      
      #Will change this if needed
      name2 = None
      
      #First of all restore the best fit values
      #for k,v in values.iteritems():
          
      #    self.minuit.values[ k ] = v
      
      #Now set the parameters under scrutiny to the current values
      
      #Since iminuit does not allow to fix parameters,
      #I am forced to create a new one (which sucks)
      
      newargs = dict( self.minuit.fitarg )
      
      newargs[ 'fix_%s' % name1 ] = True
      newargs[ '%s' % name1 ] = aa
            
      if numpy.isfinite(bb):
          
          name2 = "%s_of_%s" % ( param2, src2 )
          
          newargs[ 'fix_%s' % name2 ] = True
          newargs[ '%s' % name2 ] = bb
      
      else:
          
          #We are stepping through one param only.
          #Do nothing
          
          pass
      
      newargs['errordef'] = 0.5
      newargs['print_level'] = 0
      
      m = Minuit(self._f, **newargs )
      
      #High tolerance for speed
      m.tol = 100
      
      #mpl.warning("Running migrad")
      
      #Handle the corner case where there are no free parameters
      #after fixing the two under scrutiny
      
      #free = [k for k, v in self.contour_minuit.fixed.iteritems() if not v]
            
      if len( m.list_of_vary_param() )==0:
          
          #All parameters are fixed, just return the likelihood function
          
          if name2 is None:
              
              val = self._f(aa)
          
          else:
              
              #This is needed because the user could specify the
              #variables in reverse order
              
              myvars = [0] * 2
              myvars[ self.nameToPos[ name1 ] ] = aa
              myvars[ self.nameToPos[ name2 ] ] = bb 
          
              val = self._f(*myvars)
                    
          return val
          

      try:
        
        m.migrad()
      
      except:
            
        #In this context this is not such a big deal,
        #because we might be so far from the minimum that
        #the fit cannot converge
        
        return FIT_FAILED
      
      else:
        
        pass
        
        #print("%s -> %s" % (self.minuit.values, self.minuit.fval))
      
      #mpl.warning("Returning")
            
      return m.fval    
    
    #Do the computation
    
    if(progress):
      
      prog = ProgressBar(grid.shape[0])
      
      def wrap(args):
        prog.increase()
        return contourWorker(args)
      
      r = map( wrap, grid )
      
    else:
      
      r = map( contourWorker, grid )
    
    return a,b,numpy.array(r).reshape((a.shape[0],b.shape[0]))
