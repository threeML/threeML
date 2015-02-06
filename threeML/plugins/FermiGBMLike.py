import pyfits
import xml.etree.ElementTree as ET
import os
from threeML.plugins.gammaln import logfactorial
import numpy
from threeML.plugins.ogip import OGIPPHA
from threeML.pluginPrototype import pluginPrototype
import scipy.integrate

__instrument_name = "Fermi GBM (all detectors)"

class FermiGBMLike(pluginPrototype):
  
  def __init__(self,name,phafile,bkgfile,rspfile):
    '''
    If the input files are PHA2 files, remember to specify the spectrum number, for example:
    FermiGBMLike("GBM","spectrum.pha{2}","bkgfile.bkg{2}","rspfile.rsp{2}")
    to load the second spectrum, second background spectrum and second response.
    '''
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
    pass
    
    self.phafile              = OGIPPHA(phafile,filetype='observed')
    self.exposure             = self.phafile.getExposure()
    self.bkgfile              = OGIPPHA(bkgfile,filetype="background")
    self.response             = Response(rspfile)    
    
    #Start with an empty mask (the user will overwrite it using the
    #setActiveMeasurement method)
    self.mask                 = numpy.asarray(numpy.ones(self.phafile.getRates().shape),numpy.bool)
    
    #Get the counts for this spectrum
    self.counts               = self.phafile.getRates()[self.mask]*self.exposure
    self.bkgCounts            = self.bkgfile.getRates()[self.mask]*self.exposure
    
    self.name                 = name
  pass
    
  def setActiveMeasurements(self,*args):
    '''Set the measurements to be used during the analysis. Use as many ranges as you need,
    specified as 'emin-emax'. Energies are in keV. Example:
    
    setActiveMeasurements('10-12.5','56.0-100.0')
    
    which will set the energy range 10-12.5 keV and 56-100 keV to be used in the analysis'''
    
    #To implelemnt this we will use an array of boolean index, which will filter
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
    
    print("Now using %s channels out of %s" %(numpy.sum(self.mask),self.phafile.getRates().shape[0]))
  pass
  
  def getName(self):
    '''
    Return a name for this dataset (likely set during the constructor)
    '''
    return self.name
  pass
  
  def setModel(self,ModelManagerInstance):
    '''
    Set the model to be used in the joint minimization. Must be a ModelManager instance.
    '''
    self.modelManager         = ModelManagerInstance
    self.response.setFunction(self.modelManager.spectralModel,self.modelManager.spectralModel.integral)
  pass

  def innerFit(self):
    '''
    This is used for the profile likelihood. Keeping fixed all parameters in the
    modelManager, this method minimize the logLike over the remaining nuisance
    parameters, i.e., the parameters belonging only to the model for this
    particular detector
    '''
    #There are no nuisance parameters here
    return self.getLogLike()
  pass
  
  def getFoldedModel(self):
    #Get the folded model for this spectrum (this is the rate predicted, in cts/s)    
    return self.response.convolve()[self.mask]
  pass
  
  def getModelAndData(self):
    e1,e2                     = (self.response.ebounds[:,0],self.response.ebounds[:,1])
    return self.response.convolve()[self.mask]*self.exposure+self.bkgCounts,e1[self.mask],e2[self.mask],self.counts
  
  def getLogLike(self):
    '''
    Return the value of the log-likelihood with the current values for the
    parameters
    '''
    #Get the folded model for this spectrum (this is the rate predicted, in cts/s)    
    folded                    = self.getFoldedModel()
    
    #Model is folded+background (i.e., we assume negligible errors on the background)
    modelCounts               = folded*self.exposure + self.bkgCounts
    
    logLike                   = numpy.sum(-modelCounts+self.counts*numpy.log(modelCounts)-logfactorial(self.counts))
    
    return logLike
  pass
  
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
    if('{' in rspfile):
      tokens                    = rspfile.split("{")
      rspfile                   = tokens[0]
      rspNumber                 = int(tokens[-1].split('}')[0].replace(" ",""))
    pass
    
    #Read the response
    with pyfits.open(rspfile) as f:
      try:
        data                    = f['MATRIX',rspNumber].data
      except:
        data                    = f['SPECRESP MATRIX',rspNumber].data
      pass    
      self.matrix               = variableToMatrix(data.field('MATRIX'))
      self.ebounds              = numpy.vstack([f['EBOUNDS'].data.field("E_MIN"),f['EBOUNDS'].data.field("E_MAX")]).T
      self.mc_channels          = numpy.vstack([data.field("ENERG_LO"),data.field("ENERG_HI")]).T
    pass
    
  pass
  
  def setFunction(self,differentialFunction,integralFunction=None):
    self.differentialFunction   = differentialFunction
    if(integralFunction==None):
      def integral(x,y):
        return scipy.integrate.quad(self.differentialFunction,x,y)[0]
      self.integralFunction     = numpy.vectorize(integral,otypes=[numpy.float])
    else:
      self.integralFunction     = integralFunction
    pass
  pass
  
  def convolve(self):
    trueFluxes                  = self.integralFunction(self.mc_channels[:,0],self.mc_channels[:,1])
  
    foldedCounts                = numpy.dot(trueFluxes,self.matrix.T)
    return foldedCounts
  pass
  
  def getCountsVector(self,e1,e2):
    trueFluxes                  = self.integralFunction(self.mc_channels[:,0],self.mc_channels[:,1])
    
  pass
  
  def energyToChannel(self,energy):
    '''Finds the channel containing the provided energy. NOTE: returns the channel index (starting at zero),
    not the channel number (likely starting from 1)'''
    
    #Get the index of the first ebounds upper bound larger than energy
    try:
      idx                         = next(idx for idx, value in enumerate(self.ebounds[:,1]) if value >= energy)
    except StopIteration:
      #No values above the given energy, return the last channel
      return self.ebounds[:,1].shape[0]
    return idx
  pass
pass

def variableToMatrix(variableLengthMatrix):
  '''This take a variable length array and return it in a properly formed constant length array, to avoid some pyfits obscure bugs'''
  nrows                          = len(variableLengthMatrix)
  ncolumns                       = max([len(elem) for elem in variableLengthMatrix])
  matrix                         = numpy.zeros([ncolumns,nrows])
  for i in range(nrows):
    for j in range(ncolumns):
      try:
        matrix[j,i]                = variableLengthMatrix[i][j]
      except:
        pass
  return matrix
pass
