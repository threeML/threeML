import astropy.io.fits as pyfits
import xml.etree.ElementTree as ET
import os
from threeML.plugins.gammaln import logfactorial
import numpy
from threeML.plugins.ogip import OGIPPHA
from threeML.pluginPrototype import pluginPrototype
import scipy.integrate

import warnings

__instrument_name = "Fermi GBM (all detectors)"

class FermiGBMLike(pluginPrototype):
  
  def __init__(self,name,phafile,bkgfile,rspfile):
    '''
    If the input files are PHA2 files, remember to specify the spectrum number, for example:
    FermiGBMLike("GBM","spectrum.pha{2}","bkgfile.bkg{2}","rspfile.rsp{2}")
    to load the second spectrum, second background spectrum and second response.
    '''
    
    #Check that all file exists
    notExistant               = []
    
    if(not os.path.exists(phafile.split("{")[0])):
      notExistant.append(phafile.split("{")[0])
    
    if(not os.path.exists(bkgfile.split("{")[0])):
      notExistant.append(bkgfile.split("{")[0])
    
    if(not os.path.exists(rspfile.split("{")[0])):
      notExistant.append(rspfile.split("{")[0])
    
    if(len(notExistant)>0):
      
      for nt in notExistant:
        print("File %s does not exists!" %(nt))
      
      raise IOError("One or more input file do not exist!")
    
    
    self.phafile              = OGIPPHA(phafile,filetype='observed')
    self.exposure             = self.phafile.getExposure()
    self.bkgfile              = OGIPPHA(bkgfile,filetype="background")
    self.response             = Response(rspfile)    
    
    #Start with an empty mask (the user will overwrite it using the
    #setActiveMeasurement method)
    self.mask                 = numpy.asarray(
                                    numpy.ones(self.phafile.getRates().shape),
                                    numpy.bool)
    
    #Get the counts for this spectrum
    self.counts               = ( self.phafile.getRates()[self.mask]
                                  * self.exposure )
    
    self.bkgCounts            = ( self.bkgfile.getRates()[self.mask]
                                  * self.exposure )
    
    self.name                 = name
  pass
    
  def setActiveMeasurements(self,*args):
    '''Set the measurements to be used during the analysis. 
    Use as many ranges as you need,
    specified as 'emin-emax'. Energies are in keV. Example:
    
    setActiveMeasurements('10-12.5','56.0-100.0')
    
    which will set the energy range 10-12.5 keV and 56-100 keV to be 
    used in the analysis'''
    
    #To implelemnt this we will use an array of boolean index, 
    #which will filter
    #out the non-used channels during the logLike
    
    #Now build the mask: values for which the mask is 0 will be masked
    mask                      = numpy.zeros(self.phafile.getRates().shape)
    
    for arg in args:
      ee                      = map(float,arg.replace(" ","").split("-"))
      emin,emax               = sorted(ee)
      idx1                    = self.response.energyToChannel(emin)
      idx2                    = self.response.energyToChannel(emax)
      mask[idx1:idx2+1]       = True
    pass
    self.mask                 = numpy.array(mask,numpy.bool)
    
    self.counts               = self.phafile.getRates()[self.mask]*self.exposure
    self.bkgCounts            = self.bkgfile.getRates()[self.mask]*self.exposure
    
    print("Now using %s channels out of %s" %( numpy.sum(self.mask),
                                               self.phafile.getRates().shape[0]
                                              ) )
  pass
  
  def getName(self):
    '''
    Return a name for this dataset (likely set during the constructor)
    '''
    return self.name
  pass
  
  def setModel(self,likelihoodModel):
    '''
    Set the model to be used in the joint minimization.
    '''
    self.likelihoodModel         = likelihoodModel
    
    nPointSources                = self.likelihoodModel.getNumberOfPointSources()
    
    #This is a wrapper which iterates over all the point sources and get
    #the fluxes
    #We assume there are no extended sources, since the GBM cannot handle them
    
    def diffFlux(energies):
      
      fluxes                     = self.likelihoodModel.getPointSourceFluxes(0,energies)
      
      #If we have only one point source, this will never be executed
      for i in range(1, nPointSources):
        fluxes                  += self.likelihoodModel.getPointSourceFluxes(i,energies)
      
      return fluxes
    
    self.diffFlux                = diffFlux
    
    #The following integrates the diffFlux function using Simpson's rule
    #This assume that the intervals e1,e2 are all small, which is guaranteed
    #for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
    #energies. It also assumes that the function is smooth in the interval
    #e1 - e2 and twice-differentiable, again reasonable on small intervals for
    #decent models. It might fail for models with too sharp features, smaller
    #than the size of the monte carlo interval.
    
    def integral(e1,e2):
      
      #Simpson's rule
      
      return (e2 - e1) / 6.0 * ( self.diffFlux(e1) 
                                 + 4 * self.diffFlux( (e1+e2) / 2.0 )
                                 + self.diffFlux(e2) )
    
    self.response.setFunction( diffFlux,
                               integral)
  pass

  def innerFit(self):
  
    #There are no nuisance parameters here (at least for now)
    
    pass
  
  def getFoldedModel(self):
    
    #Get the folded model for this spectrum 
    #(this is the rate predicted, in cts/s)    
    
    return self.response.convolve()[self.mask]
  
  def getModelAndData(self):
    
    e1,e2                     = (self.response.ebounds[:,0],
                                 self.response.ebounds[:,1])
    
    return ( self.response.convolve()[self.mask] * self.exposure 
           + self.bkgCounts, 
             e1[self.mask],
             e2[self.mask],
             self.counts )
  
  def getLogLike(self):
    '''
    Return the value of the log-likelihood with the current values for the
    parameters
    '''
    
    #Get the folded model for this spectrum (this is the rate predicted, 
    #in cts/s)    
    folded                    = self.getFoldedModel()
    
    #Model is folded+background (i.e., we assume negligible errors on the 
    #background)
    modelCounts               = folded * self.exposure + self.bkgCounts
    
    logLike                   = numpy.sum(- modelCounts 
                                          + self.counts * numpy.log(modelCounts)
                                          - logfactorial(self.counts) )
    
    return logLike
      
  def getNuisanceParameters(self):
    '''
    Return a list of nuisance parameter names. Return an empty list if there
    are no nuisance parameters
    '''
    return []
  pass
pass

class Response(object):
  def __init__(self,rspfile):
    
    rspNumber                   = 1
    
    #Check if we are dealing with a .rsp2 file (containing more than
    #one response). This is checked by looking for the syntax 
    # [responseFile]{[responseNumber]}
    
    if('{' in rspfile):
    
      tokens                    = rspfile.split("{")
      rspfile                   = tokens[0]
      rspNumber                 = int(tokens[-1].split('}')[0].replace(" ",""))
    
    
    #Read the response
    with pyfits.open(rspfile) as f:
      
      try:
        
        #GBM typical response
        data                    = f['MATRIX',rspNumber].data
      
      except:
        #Other detectors might use the SPECRESP MATRIX name instead
        #Note that here we are not catching any exception, because
        #we have to fail if we cannot read the matrix
        
        data                    = f['SPECRESP MATRIX',rspNumber].data
      
      #Sometimes .rsp files contains a weird format featuring variable-length
      #arrays. Historically those confuse pyfits quite a lot, so we ensure 
      #to transform them into standard numpy matrices to avoid issues
      
      self.matrix               = variableToMatrix(data.field('MATRIX'))
      
      self.ebounds              = numpy.vstack([ f['EBOUNDS'].data.field("E_MIN"),
                                                 f['EBOUNDS'].data.field("E_MAX") ]).T
      
      self.mc_channels          = numpy.vstack([ data.field("ENERG_LO"),
                                                 data.field("ENERG_HI") ]).T
    
  def setFunction(self, differentialFunction, integralFunction=None):
    '''
    Set the function to be used for the convolution
    '''
    
    self.differentialFunction   = differentialFunction
    
    if(integralFunction==None):
      
      #This should never happen in 3ML, but we keep this functionality for
      #other uses. Print a warning anyway
      
      warnings.warn("Using the slow numerical integration in the GBM plugin!", 
                     RuntimeWarning)
      
      def integral(x,y):
        return scipy.integrate.quad(self.differentialFunction,x,y)[0]
      
      #NB: vectorize is super slow!
      self.integralFunction     = numpy.vectorize(integral,otypes=[numpy.float])
    
    else:
      
      self.integralFunction     = integralFunction
    
  
    
  def convolve(self):
    
    trueFluxes                  = self.integralFunction( self.mc_channels[:,0],
                                                         self.mc_channels[:,1] )
  
    foldedCounts                = numpy.dot( trueFluxes, self.matrix.T )
    
    return foldedCounts
  
  def getCountsVector(self,e1,e2):
    
    trueFluxes                  = self.integralFunction(self.mc_channels[:,0],self.mc_channels[:,1])
  
  def energyToChannel(self,energy):
    
    '''Finds the channel containing the provided energy. 
    NOTE: returns the channel index (starting at zero),
    not the channel number (likely starting from 1)'''
    
    #Get the index of the first ebounds upper bound larger than energy
    
    try:
    
      idx                         = next( idx for idx,
                                          value in enumerate(self.ebounds[:,1]) 
                                          if value >= energy )
    
    except StopIteration:
      
      #No values above the given energy, return the last channel
      return self.ebounds[:,1].shape[0]
    
    return idx
  

def variableToMatrix(variableLengthMatrix):
  '''This take a variable length array and return it in a 
  properly formed constant length array, to avoid some pyfits obscure bugs'''
  
  nrows                          = len(variableLengthMatrix)
  ncolumns                       = max([len(elem) for elem in variableLengthMatrix])
  matrix                         = numpy.zeros([ncolumns,nrows])
  
  for i in range(nrows):
    for j in range(ncolumns):
      
      try:
      
        matrix[j,i]                = variableLengthMatrix[i][j]
      
      except:
        
        #This happens when the i,j element does not exist, which is not an error
        #We will just leave it to zero
        pass
        
  return matrix
