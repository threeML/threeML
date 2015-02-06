import numpy
import collections
from Parameter import Parameter
import fancyDisplay
from IPython.display import display, Latex, HTML
import math
import scipy.integrate
import operator
import numexpr

keVtoErg                 = 1.60217657E-9

class SpectralModel(object):  
  def __getitem__(self,argument):
    return self.parameters[argument]
  
  def __repr__(self):
    print("Spectral model: %s" %(self.functionName))
    print("Formula:\n")
    display(Latex(self.formula))
    print("")
    print("Current parameters:\n")
    table                    = fancyDisplay.HtmlTable(8)
    table.addHeadings("Name","Value","Minimum","Maximum","Delta","Status","Unit","Prior")
    for k,v in self.parameters.iteritems():
      if(v.isFree()):
        ff                   = "free"
      else:
        ff                   = "fixed"
      pass
      table.addRow(v.name,v.value,v.minValue,v.maxValue,v.delta,ff,v.unit,v.prior.getName())
    pass
    display(HTML(table.__repr__()))
    
    return ''
  pass
  
  #You MUST override this
  def __call__(self):
    raise NotImplemented("The method __call__ has not been implemented. This is a bug")
  
  #If there is an analytical solution to the integral of the function,
  #override these methods: the analytical way is always A LOT faster
  #than numerical integration
  def photonFlux(self,e1,e2):
    return scipy.integrate.quad(self.__call__,e1,e2,epsabs=0.01,epsrel=1e-4)[0]
  pass
  
  def energyFlux(self,e1,e2):
    def eF(e):
      return e*self.__call__(e)
    
    return scipy.integrate.quad(eF,e1,e2,epsabs=0,epsrel=1e-3)[0]*keVtoErg
  pass
  
  #The following methods define the aritmetical operations on model, so
  #that the user can define "composite" models
  def __add__(self,b):
    return CompositeModel(self,b,operator.add)
  
  def __radd__(self,b):
    return CompositeModel(self,b,operator.add)
  
  def __sub__(self,b):
    return CompositeModel(self,b,operator.sub)
  
  def __rsub__(self,b):
    return CompositeModel(self,b,operator.sub)
  
  def __div__(self,b):
    return CompositeModel(self,b,operator.div)

  def __rdiv__(self,b):
    return CompositeModel(self,b,operator.div)
  
  def __mul__(self,b):
    return CompositeModel(self,b,operator.mul)

  def __rmul__(self,b):
    return CompositeModel(self,b,operator.mul)
  
pass

#The following class represent the combination of two
#models with an aritmetical operation. For example,
#CompositeModel(model1,model2,operator.add) will return
#a class which will act like a (model1+model2) model.

class CompositeModel(SpectralModel):
  def __init__(self,one,two,thisOperator):
    if type(one) in (int,float):
      self.oneCall            = lambda x:float(one)
    elif issubclass(type(one),(SpectralModel,CompositeModel)):
      self.oneCall            = one.__call__
    else:
      raise NotImplementedError("Cannot execute operation with operand of type %s" %(type(one)))
    
    if type(two) in (int,float):
      self.twoCall            = lambda x:float(two)
    elif issubclass(type(two),(SpectralModel,CompositeModel)):
      self.twoCall            = two.__call__
    else:
      raise NotImplementedError("Cannot execute operation with operand of type %s" %(type(one)))
    
    self.operator             = thisOperator
    
    #Now create a parameters dictionary pointing to the
    #parameters in one and two
    self.parameters           = collections.OrderedDict()
    
    for component in [one,two]:
      if type(component) in (int,float):
        continue
      for k in component.parameters.keys():
        if(k in self.parameters.keys()):
          tokens                = k.split("_")
          if(len(tokens)==1):
            newKey              = k+"_1"
            tokens              = newKey.split("_")
          pass
          oldNum              = int(tokens[1])
          for i in range(oldNum,1000):
            newKey            = tokens[0]+"_%s" %(i)
            if(newKey not in self.parameters.keys()):
              break
            pass
          pass
        else:
          newKey                = k
        pass
        
        self.parameters[newKey] = component.parameters[k]
      pass
    pass
        
    self.formula              = "(print the components by themselves to know their formulae)"
    
    operatorNames             = {operator.add: ' + ', operator.div: ' / ', operator.mul: ' * ', operator.sub: ' - '}
    
    names                     = []
    for component in [one,two]:
      if type(component) in (int,float):
        names.insert(0,"%s" %(component))
      else:
        names.append(component.functionName)
    pass
    
    self.functionName       = "( " + operatorNames[self.operator].join(names) + " )"
    
    integrals               = []
    for component in [one,two]:
      if type(component) in (int,float):
        integrals.append(lambda e1,e2: float(component))
      else:
        integrals.append(component.integral)
      pass
    pass
    self.integral           = lambda e1,e2: integrals[0](e1,e2)+integrals[1](e1,e2)
  pass
  
  def __call__(self,x):
    return self.operator(self.oneCall(x),self.twoCall(x))
  pass
pass


class PowerLaw(SpectralModel):
  def __init__(self):
    self.functionName        = "Powerlaw"
    self.formula             = r'\begin{equation}f(E) = A E^{\gamma}\end{equation}'
    self.parameters          = collections.OrderedDict()
    self.parameters['gamma'] = Parameter('gamma',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
    self.parameters['A']     = Parameter('A',1.0,1e-10,1e10,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
    self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True)
    
    self.ncalls              = 0
    
    def integral(e1,e2):
      a                      = self.parameters['gamma'].value
      piv                    = self.parameters['Epiv'].value
      norm                   = self.parameters['A'].value
      
      if(a!=-1):
        def f(energy):
          
          return self.parameters['A'].value * math.pow(energy/piv,a+1)/(a+1)
      else:
        def f(energy):
          return self.parameters['A'].value * math.pow(energy/piv)
      return f(e2)-f(e1)
    self.integral            = integral
  pass
  
  def __call__(self,energy):
    self.ncalls             += 1
    piv                      = self.parameters['Epiv'].value
    norm                     = self.parameters['A'].value
    gamma                    = self.parameters['gamma'].value
    return numpy.maximum( numexpr.evaluate("norm * (energy/piv)**gamma"), 1e-30)
  pass
  
  def photonFlux(self,e1,e2):
    return self.integral(e1,e2)
  
  def energyFlux(self,e1,e2):
    a                        = self.parameters['gamma'].value
    piv                      = self.parameters['Epiv'].value
    if(a!=-2):
      def eF(e):
        return numpy.maximum(self.parameters['A'].value * numpy.power(e/piv,2-a)/(2-a),1e-30)
    else:
      def eF(e):
        return numpy.maximum(self.parameters['A'].value * numpy.log(e/piv),1e-30)
    pass
    
    return (eF(e2)-eF(e1))*keVtoErg
pass

from scipy import weave
from scipy.weave import converters

class wPowerLaw(PowerLaw):
  def __call__(self,energy):
    a                         = self.parameters['A'].value
    alpha                     = self.parameters['gamma'].value
    try:
      n                       = energy.shape[0]
    except:
      n                       = 1
    pass
    result                    = numpy.array([0]*n,dtype=float)
    code="""
    int i;
    for(i=0;i<n;i++)
    {
      result(i) = a*pow(energy(i),alpha);
    }
    """
    weave.inline(code,['a','alpha','n','energy','result'],type_converters=converters.blitz)
    return result
  pass
pass 




class Band(SpectralModel):
  def __init__(self):
    self.functionName       = "Band function [Band et al. 1993]"
    self.formula            = r'''
    \[f(E) = \left\{ \begin{eqnarray}
K \left(\frac{E}{100 \mbox{ keV}}\right)^{\alpha} & \exp{\left(\frac{-E}{E_{c}}\right)} & \mbox{ if } E < (\alpha-\beta)E_{c} \\
    K \left[ (\alpha-\beta)\frac{E_{c}}{100}\right]^{\alpha-\beta}\left(\frac{E}{100 \mbox{ keV}}\right)^{\beta} & \exp{(\beta-\alpha)} & \mbox{ if } E \ge (\alpha-\beta)E_{c}
    \end{eqnarray}
    \right.
    \]
    '''
    self.parameters         = collections.OrderedDict()
    self.parameters['alpha'] = Parameter('alpha',-1.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
    self.parameters['beta']  = Parameter('beta',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
    self.parameters['E0']    = Parameter('E0',500,10,1e5,50,fixed=False,nuisance=False,dataset=None,unit='keV')
    self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
    
    def integral(e1,e2):
      return self((e1+e2)/2.0)*(e2-e1)
    self.integral            = integral
    
  pass
  
  def __call__(self,e):
    #The input e can be either a scalar or an array
    #The following will generate a wrapper which will
    #allow to treat them in exactly the same way,
    #as they both were arrays, while keeping the output
    #in line with the input: if e is a scalar, the output
    #will be a scalar; if e is an array, the output will be an array
    energies                 = numpy.array(e,ndmin=1,copy=False)
    alpha                    = self.parameters['alpha'].value
    beta                     = self.parameters['beta'].value
    E0                       = self.parameters['E0'].value
    K                        = self.parameters['K'].value
    
    out                      = numpy.zeros(energies.flatten().shape[0])
    idx                      = (energies < (alpha-beta)*E0)
    nidx                     = ~idx
    out[idx]                 = numpy.maximum(K*numpy.power(energies[idx]/100.0,alpha)*numpy.exp(-energies[idx]/E0),1e-30)
    out[nidx]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(energies[nidx]/100.0,beta),1e-30)
    
    #This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
    out                      = numpy.nan_to_num(out)
    if(out.shape[0]==1):
      return out[0]
    else:
      return out
  pass    
  
pass

class wBand(Band):
  def __call__(self,e):
    energies                 = numpy.array(e,ndmin=1,copy=False)
    alpha                    = self.parameters['alpha'].value
    beta                     = self.parameters['beta'].value
    E0                       = self.parameters['E0'].value
    K                        = self.parameters['K'].value
    
    n                        = energies.shape[0]
    #result                    = numpy.zeros(n)
    code="""
    double *result = (double *)malloc(n*sizeof(double));
    int i;
    double ee,E,E100,ee100;
    ee = (alpha+beta)*E0;
    ee100 = ee/1000.0;
    for(i=0;i<n;i++)
    {
      E = energies(i);
      E100 = E/1000.0;
      if(E < ee) {
        result[i] = K*pow(E100,alpha)*exp(-E/E0);
      } else {
        result[i] = K*pow(ee100,alpha-beta)*exp(beta-alpha)*pow(E100,beta);
      }
    }
    return_val = result;
    """
    return weave.inline(code,['energies','alpha','beta','E0','K','n'],type_converters=converters.blitz)
  
  pass
pass

class LogParabola(SpectralModel):
  def __init__(self):
    #(E/pivotE)**(-alpha-beta*log(E/pivotE))
    self.functionName        = "LogParabola"
    self.formula             = r'\begin{equation}f(E) = A E^{\gamma+\beta \log(E)}\end{equation}'
    self.parameters          = collections.OrderedDict()
    self.parameters['gamma'] = Parameter('gamma',-1.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
    self.parameters['beta'] = Parameter('beta',-0.5,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
    self.parameters['A']     = Parameter('A',1.0,1e-10,1e10,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
    self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True)
    
    self.ncalls              = 0
    
    def integral(e1,e2):
      return self((e1+e2)/2.0)*(e2-e1)
    self.integral            = integral
  pass
  
  def __call__(self,energy):
    self.ncalls             += 1
    piv                      = self.parameters['Epiv'].value
    gamma                = self.parameters['gamma'].value
    beta                    = self.parameters['beta'].value
    return numpy.maximum(self.parameters['A'].value * numpy.power(energy/piv,gamma+(beta*numpy.log10(energy/piv))),1e-35)
  pass
  
  def photonFlux(self,e1,e2):
    return self.integral(e1,e2)
  
  #def energyFlux(self,e1,e2):
  #  a                        = self.parameters['gamma'].value
  #  piv                      = self.parameters['Epiv'].value
  #  if(a!=-2):
  #    def eF(e):
  #      return numpy.maximum(self.parameters['A'].value * numpy.power(e/piv,2-a)/(2-a),1e-30)
  #  else:
  #    def eF(e):
  #      return numpy.maximum(self.parameters['A'].value * numpy.log(e/piv),1e-30)
  #  pass
    
   # return (eF(e2)-eF(e1))*keVtoErg
pass



class Madness(SpectralModel):
  def __init__(self,model,nBreaks=100,emin=1.0,emax=1e9,**kwargs):
    self.functionName        = "Madness"
    self.formula             = r'\begin{equation}f(E) = K_{0} E^{\alpha}\end{equation}'
    self.parameters          = collections.OrderedDict()
    
    self.nBreaks             = int(nBreaks)
    self.energyBreaks        = numpy.logspace(numpy.log10(emin),numpy.log10(emax),self.nBreaks+2)[1:-1]
    
    for k,v in kwargs.iteritems():
      if(k.lower()=='breaks'):
        print("Using user-provided breaks")
        self.energyBreaks    = numpy.array(v)
        self.nBreaks         = self.energyBreaks.shape[0]
      pass
    pass
    
    #Choose the pivot energies equal to the breaks, for convenience. Use emin as pivot energy
    #for the first branch, before the first break
    self.pivotEnergies       = numpy.concatenate([[emin],self.energyBreaks])
        
    #Initial values for alphas: a curved spectrum starting with -1
    #and ending with -2.5, like GRBs do when modeled with a Band function
    alphas                   = numpy.linspace(-2.5,-1,self.nBreaks+1)[::-1]
            
    #Add all the alphas to the parameters list
    for i in xrange(self.nBreaks+1):
      #Normalization parameter
      thisNorm                 = 'K_%s' %i
      pv                       = model(self.pivotEnergies[i])
      self.parameters[thisNorm] = Parameter(thisNorm,pv,pv/100,pv*100,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
      thisAlpha                = 'alpha_%s' %i
      self.parameters[thisAlpha]   = Parameter(thisAlpha,alphas[i],-10,10,0.1,fixed=True,nuisance=False,dataset=None)      
    pass
    
    def integral(e1,e2):
      #I'm too lazy now to write the analytical expression
      return self((e1+e2)/2.0)*(e2-e1)
      
    self.integral            = integral
    
    self.normalizations      = numpy.zeros(self.nBreaks+2)
    
    self._computeNormalizations()
  pass
  
  def _computeNormalizations(self):
    #Use a generator instead of a list to gain speed
    generator                = (x.value for x in self.parameters.values()[1::2])
    self.indexes             = numpy.fromiter(generator,float)
    indexDiff                = self.indexes[:-1]-self.indexes[1:]
              
    self.normalizations[0]   = self.parameters['K_0'].value
    self.normalizations[1:-1]  = (numpy.power(self.energyBreaks,indexDiff)*
                                  numpy.power(self.pivotEnergies[1:],self.indexes[1:])
                                  /
                                  numpy.power(self.pivotEnergies[:-1],self.indexes[:-1]))
    self.normalizations[-1]  = 1.0
    
    #This compute the cumulative product of the array
    #(i.e., the first elements is a0, the second a0*a1,
    #the third a0*a1*a2, and so on...)
    self.products            = numpy.cumprod(self.normalizations)
  pass
  
  def __call__(self,energy):
    self._computeNormalizations()
    
    #Make this always an array
    energies                 = numpy.array(energy,ndmin=1,copy=False)
    energies.sort()
    
    #This find the indexes of the places in which the breaks should be inserted
    #to keep the order of the array. In other words, for each break finds the index of
    #the first value larger or equal to the break energy
    indexes                  = numpy.searchsorted(energies,self.energyBreaks)
    indexes                  = numpy.concatenate(([0],indexes,[energies.shape[0]]))
    
    results                  = numpy.empty(energies.shape)
    
    for i in xrange(self.nBreaks+1):
      i1,i2                  = (indexes[i],indexes[i+1])
      thisNorm               = self.parameters['K_%s' %i].value
      thisPivot              = self.pivotEnergies[i]
      results[i1:i2]         = thisNorm*numpy.power(energies[i1:i2]/thisPivot,self.indexes[i])
    pass   
    
    return numpy.maximum(results,1e-30)
  pass
  
pass

class ManyBrokenPowerlaws(SpectralModel):
  def __init__(self,nBreaks=100,emin=1.0,emax=1e9,**kwargs):
    self.functionName        = "ManyBrokenPowerlaws"
    self.formula             = r'\begin{equation}f(E) = K_{0} E^{\alpha}\end{equation}'
    self.parameters          = collections.OrderedDict()
    
    self.nBreaks             = int(nBreaks)
    self.energyBreaks        = numpy.logspace(numpy.log10(emin),numpy.log10(emax),self.nBreaks+2)[1:-1]
    
    for k,v in kwargs.iteritems():
      if(k.lower()=='breaks'):
        print("Using user-provided breaks")
        self.energyBreaks    = numpy.array(v)
        self.nBreaks         = self.energyBreaks.shape[0]
      pass
    pass
    
    #Choose the pivot energies equal to the breaks, for convenience. Use emin as pivot energy
    #for the first branch, before the first break
    self.pivotEnergies       = numpy.concatenate([[emin],self.energyBreaks])
    
    #Initial values for alphas: a curved spectrum starting with -1
    #and ending with -2.5, like GRBs do when modeled with a Band function
    alphas                   = numpy.linspace(-2.5,-1,self.nBreaks+1)[::-1]

    #Normalization parameter
    self.parameters['K']     = Parameter('K',1.0,1e-6,1e3,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
        
    #Add all the alphas to the parameters list
    for i in xrange(self.nBreaks+1):
      thisName               = 'alpha_%s' %i
      self.parameters[thisName]   = Parameter(thisName,alphas[i],-10,10,0.1,fixed=False,nuisance=False,dataset=None)      
    pass
    
    def integral(e1,e2):
      #I'm too lazy now to write the analytical expression
      return self((e1+e2)/2.0)*(e2-e1)
      
    self.integral            = integral
    
    self.normalizations      = numpy.zeros(self.nBreaks+2)
    
    self._computeNormalizations()
  pass
  
  def _computeNormalizations(self):
    #Use a generator instead of a list to gain speed
    generator                = (x.value for x in self.parameters.values()[1:])
    self.indexes             = numpy.fromiter(generator,float)
    indexDiff                = self.indexes[:-1]-self.indexes[1:]
              
    self.normalizations[0]   = self.parameters['K'].value
    self.normalizations[1:-1]  = (numpy.power(self.energyBreaks,indexDiff)*
                                  numpy.power(self.pivotEnergies[1:],self.indexes[1:])
                                  /
                                  numpy.power(self.pivotEnergies[:-1],self.indexes[:-1]))
    self.normalizations[-1]  = 1.0
    
    #This compute the cumulative product of the array
    #(i.e., the first elements is a0, the second a0*a1,
    #the third a0*a1*a2, and so on...)
    self.products            = numpy.cumprod(self.normalizations)
  pass
  
  def __call__(self,energy):
    self._computeNormalizations()
    
    #Make this always an array
    energies                 = numpy.array(energy,ndmin=1,copy=False)
    energies.sort()
    
    #This find the indexes of the places in which the breaks should be inserted
    #to keep the order of the array. In other words, for each break finds the index of
    #the first value larger or equal to the break energy
    indexes                  = numpy.searchsorted(energies,self.energyBreaks)
    indexes                  = numpy.concatenate(([0],indexes,[energies.shape[0]]))
    
    results                  = numpy.empty(energies.shape)
    
    for i in xrange(self.nBreaks+1):
      i1,i2                  = (indexes[i],indexes[i+1])
      thisNorm               = self.products[i]
      thisPivot              = self.pivotEnergies[i]
      results[i1:i2]         = thisNorm*numpy.power(energies[i1:i2]/thisPivot,self.indexes[i])
    pass   
    
    return numpy.maximum(results,1e-30)
  pass
  
pass


class ManyLogparabolas(SpectralModel):
  def __init__(self,nBreaks=30,emin=5.0,emax=1e9,**kwargs):
    self.functionName        = "ManyLogparabolas"
    self.formula             = r'\begin{equation}f(E) = K_{0} E^{\alpha}\end{equation}'
    self.parameters          = collections.OrderedDict()
    
    self.nBreaks             = int(nBreaks)
    self.energyBreaks        = numpy.logspace(numpy.log10(emin),numpy.log10(emax),self.nBreaks+2)[1:-1]
    
    for k,v in kwargs.iteritems():
      if(k.lower()=='breaks'):
        print("Using user-provided breaks")
        self.energyBreaks    = numpy.array(v)
        self.nBreaks         = self.energyBreaks.shape[0]
      pass
    pass
    
    #Pivot energies: these are used to avoid the meaning of beta to change
    #over the energy. Without these, to get the same curvature you will have
    #to use different betas depending on where you are in the energy range
    self.pivotEnergies       = numpy.concatenate([[1.0],self.energyBreaks])
        
    #Initial values for alphas: a curved spectrum starting with -1
    #and ending with -2.5, like GRBs do when modeled with a Band function
    alphas                   = numpy.linspace(-2.5,-1,self.nBreaks+1)[::-1]
    betas                    = numpy.random.uniform(-1,1,self.nBreaks+1)[::-1]
    
    #Normalization parameter
    self.parameters['K']     = Parameter('K',10.0,1e-5,1e6,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
        
    #Add all the alphas
    for i in xrange(self.nBreaks+1):
      thisName               = 'alpha_%s' %i
      self.parameters[thisName]   = Parameter(thisName,alphas[i],-8,8,0.1,fixed=False,nuisance=False,dataset=None)
      thisName               = 'beta_%s' %i
      self.parameters[thisName]   = Parameter(thisName,betas[i],-5,5,0.1,fixed=False,nuisance=False,dataset=None)
    pass
    
    def integral(e1,e2):
      #I'm too lazy now to write the analytical expression
      return self((e1+e2)/2.0)*(e2-e1)
      
    self.integral            = integral
    
    self.normalizations      = numpy.zeros(self.nBreaks+2)
    
    self._computeNormalizations()
  pass
  
  def _computeNormalizations(self):
    
    #Use a generator instead of a list to gain speed
    generator1               = (x.value for x in self.parameters.values()[1::2])
    self.alphas              = numpy.fromiter(generator1,float)
    #alphasDiff               = self.alphas[:-1]-self.alphas[1:]
    
    generator2               = (x.value for x in self.parameters.values()[2::2])
    self.betas               = numpy.fromiter(generator2,float)
    #betasDiff                = self.betas[:-1]-self.betas[1:]
    
    #bLogEpivot               = self.betas*self.logPivotEnergies
    #bLogEpivotDiff           = bLogEpivot[1:]-bLogEpivot[:-1]

        
    self.normalizations[0]   = self.parameters['K'].value
    self.normalizations[1:-1]  = (self._logP(self.energyBreaks,self.alphas[:-1],self.betas[:-1],self.pivotEnergies[:-1])/
                                  self._logP(self.energyBreaks,self.alphas[1:],self.betas[1:],self.pivotEnergies[1:])
                                  )
    self.normalizations[-1]  = 1.0
    
    #This compute the cumulative product of the array
    #(i.e., the first elements is a0, the second a0*a1,
    #the third a0*a1*a2, and so on...)
    self.products            = numpy.cumprod(self.normalizations)
  pass
  
  def _logP(self,energies,alphas,betas,pivotEnergies):
    return numpy.power(energies/pivotEnergies,alphas+betas*numpy.log10(energies/pivotEnergies))
  
  def __call__(self,energy):
    self._computeNormalizations()
    
    #Make this always an array
    energies                 = numpy.array(energy,ndmin=1,copy=False)
    energies.sort()
    
    #This find the indexes of the places in which the breaks should be inserted
    #to keep the order of the array. In other words, for each break finds the index of
    #the first value larger or equal to the break energy
    indexes                  = numpy.searchsorted(energies,self.energyBreaks)
    indexes                  = numpy.concatenate(([0],indexes,[energies.shape[0]]))
    
    results                  = numpy.empty(energies.shape)
    
    for i in xrange(self.nBreaks+1):
      i1,i2                  = (indexes[i],indexes[i+1])
      thisNorm               = self.products[i]
      Ee                     = energies[i1:i2]
      results[i1:i2]         = thisNorm*self._logP(Ee,self.alphas[i],self.betas[i],self.pivotEnergies[i])
    pass   
    
    return numpy.maximum(results,1e-30)
  pass
  
pass

