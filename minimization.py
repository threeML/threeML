import numpy
import itertools

try:
  import ROOT
except:
  hasROOT                     = False
  print("You don't have ROOT installed (or pyROOT configured). You cannot use the MINUIT minimizer.")
else:
  hasROOT                     = True

try:
  import scipy.optimize
  import scipy.misc
  import numdifftools
except:
  hasScipy                     = False
  print("You don't have scipy properly installed and/or numdifftools. You cannot use the scipy minimizer.")
else:
  hasScipy                     = True
  
class Minimizer(object):
  def __init__(self,function,parameters,ftol=1e-3,verbosity=1):
    #Save all the input as members of the class for debugging purposes
    self.function             = function
    self.parameters           = parameters
    self.Npar                 = len(self.parameters.keys())
    self.ftol                 = ftol
    self.verbosity            = verbosity
  pass
  
  def minimize(self):
    raise NotImplemented("This is the method of the base class. Must be implemented by the actual minimizer")
  pass
pass

class MinuitMinimizer(Minimizer):
  def __init__(self,function,parameters,ftol=1e-1,verbosity=0):
    super(MinuitMinimizer, self).__init__(function,parameters,ftol,verbosity)
    
    #Setup the minimizer algorithm
    self.functor              = FuncWrapper(self.function,self.Npar)
    self.minimizer            = ROOT.Math.Factory.CreateMinimizer("Minuit","Minimize")
    self.minimizer.Clear()
    self.minimizer.SetMaxFunctionCalls(1000)
    self.minimizer.SetTolerance(0.1)
    self.minimizer.SetPrintLevel(self.verbosity)
    #self.minimizer.SetStrategy(0)
    
    self.minimizer.SetFunction(self.functor)
    
    for i,par in enumerate(self.parameters.values()):
      self.minimizer.SetLimitedVariable(i,par.name,par.value,par.delta,par.minValue,par.maxValue)
    pass
  pass
  
  def minimize(self,minos=False,printout=True,verbosity=0):
    self.minimizer.SetPrintLevel(int(verbosity))
    
    self.minimizer.Minimize()
    
    #This improves on the error computation
    #self.minimizer.Hesse()
    
    xs                         = numpy.array(map(lambda x:x[0],zip(self.minimizer.X(),range(self.Npar))))
    
    if(minos):
      #Get the errors
      xserr                      = []
      for i in range(xs.shape[0]):
        minv                     = ROOT.Double(0)
        maxv                     = ROOT.Double(0)
        self.minimizer.GetMinosError(i,minv,maxv)
        xserr.append([minv,maxv])
      pass
    else:
     xserr                      = numpy.array(map(lambda x:x[0],zip(self.minimizer.Errors(),range(self.Npar))))
    pass
    
    if(printout):
      print("Minimum is: %s" %(self.functor(xs)))
      print("Values for the parameters at the minimum are:")
      for name,value,error in zip(self.paramNames,xs,xserr):
        print("%-20s = %s [%s,%s]" %(name,value,error[0],error[1]))
      pass
    pass
    
    return xs,xserr,self.functor(xs)
  pass

if(hasROOT):
  class FuncWrapper( ROOT.TPyMultiGenFunction ):
      def __init__(self,function,dimensions):
          ROOT.TPyMultiGenFunction.__init__( self, self )
          self.function         = function
          self.dimensions       = dimensions
      pass
      
      def NDim( self ):
          return self.dimensions
      pass
      
      def DoEval( self, args ):
          return self.function(args)
      pass
  pass
pass

class ScipyMinimizer(Minimizer):
  def __init__(self,function,parameters,ftol=1e-12,verbosity=1):
    super(ScipyMinimizer, self).__init__(function,parameters,ftol,verbosity)
    
    #setup the scipy minimizer
    self.x0                   = map(lambda x:x.value,self.parameters.values())
    self.bounds               = map(lambda x:[x.minValue,x.maxValue],self.parameters.values())
    
  pass
  
  def minimize(self,minos=False,printout=False,verbosity=0):
    res                       = scipy.optimize.minimize(self.function,self.x0,bounds=self.bounds,
                                                        tol=self.ftol,options={'maxiter': 1000000,'disp': True})
    
    bestFitValues             = res['x']    
    
    logL                      = res['fun']    
    return bestFitValues,numpy.abs(bestFitValues*0.05),logL
  pass  
pass

try:
  import nlopt
except:
  pass
else:
  class BOBYQAMinimizer(Minimizer):
    def __init__(self,function,parameters,ftol=1e-5,verbosity=1):
      super(BOBYQAMinimizer, self).__init__(function,parameters,ftol,verbosity)
      
      #setup the bobyqa minimizer
      self.x0                   = map(lambda x:x.value,self.parameters.values())
      self.lowerBounds          = map(lambda x:x.minValue,self.parameters.values())
      self.upperBounds          = map(lambda x:x.maxValue,self.parameters.values())
      self.steps                = map(lambda x:x.delta,self.parameters.values())
      self.objectiveFunction    = function
      
      def wrapper(x,grad):
        if grad.size > 0:
          print("This won't ever happen, since BOBYQA does not use derivatives")
        return self.objectiveFunction(x)
      pass
      self.wrapper              = wrapper
            
      self.bob                  = nlopt.opt(nlopt.LN_BOBYQA, self.Npar)
      self.bob.set_min_objective(self.wrapper)
      self.bob.set_ftol_abs(ftol)
      #Stop if the value of all the parameter change by less than 1%
      self.bob.set_xtol_rel(0.001)
      self.bob.set_initial_step(self.steps)
      
      self.bob.set_lower_bounds(self.lowerBounds)
      self.bob.set_upper_bounds(self.upperBounds)
      
      #self.globalOptimizer      = nlopt.opt(nlopt.G_MLSL_LDS,self.Npar)
      #self.globalOptimizer.set_local_optimizer(self.bob)

      
    pass
    
    def minimize(self,minos=False,printout=False,verbosity=0):
      res                       = self.bob.optimize(self.x0)
      opt_val                   = self.bob.last_optimum_value()
      result                    = self.bob.last_optimize_result()
      if(result < 0):
        raise RuntimeError("Bob run into problems during the minimization (BOBYQA method failed with code %s)." %(result))
      pass
      return res,numpy.abs(res*0.05),opt_val
    pass  
  pass

pass

def RosenBrock(xx):
  x                           = xx[0];
  y                           = xx[1];
  tmp1                        = y-x*x;
  tmp2                        = 1-x;
  return 100*tmp1*tmp1+tmp2*tmp2
pass

def NumericalMinimizationTest():
    minimizer                  = ROOT.Math.Factory.CreateMinimizer("Minuit","Migrad")
    minimizer.SetMaxFunctionCalls(1000000)
    minimizer.SetMaxIterations(100000)
    minimizer.SetTolerance(0.001)
    minimizer.SetPrintLevel(10)
    
    rose                       = FuncWrapper(RosenBrock,2)

    step                       = [0.01,0.01]
    variable                   = [-1.0,1.2]

    minimizer.SetFunction(rose)

    # Set the free variables to be minimized!
    minimizer.SetVariable(0,"x",variable[0], step[0])
    minimizer.SetVariable(1,"y",variable[1], step[1])

    minimizer.Minimize()
    
    xs                         = minimizer.X()
    
    #Get the errors
    xserr                      = []
    for i in range(len(variable)):
      minv                     = ROOT.Double(0)
      maxv                     = ROOT.Double(0)
      minimizer.GetMinosError(i,minv,maxv)
      xserr.append([minv,maxv])
    #xserr                      = minimizer.Errors()
    
    print("Minimum: f(%4.3f [%4.3f,%4.3f], %4.3f [%4.3f,%4.3f]): %4.3f" %(xs[0],xserr[0][0],xserr[0][1],
                                                                          xs[1],xserr[1][0],xserr[1][1],rose(xs)))
