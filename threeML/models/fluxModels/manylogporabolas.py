from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import numpy
import math
import scipy.integrate
import operator

import collections


class ManyLogParabolas(SpectralModel):
  def setup(self,nBreaks=30,emin=5.0,emax=1e9,**kwargs):
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

