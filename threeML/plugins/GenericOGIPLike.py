import astropy.io.fits as pyfits
import xml.etree.ElementTree as ET
import os

from threeML.plugins.gammaln import logfactorial
from threeML.io import fileUtils

import numpy
from threeML.plugins.ogip import OGIPPHA
from threeML.plugin_prototype import PluginPrototype
from threeML.models.Parameter import Parameter
from threeML.minimizer import minimization
from threeML.utils import Binner
import scipy.integrate

import warnings
import collections

import matplotlib.pyplot as plt
from matplotlib import gridspec


__instrument_name = "All OGIP-compliant instruments"

class GenericOGIPLike(PluginPrototype):
  
  def __init__(self, name, phafile, bkgfile, rspfile, arffile=None):
    
    self.name                 = name
    
    #Check that all file exists
    notExistant               = []
    
    inputFiles = [ phafile, bkgfile, rspfile ]
    
    for i in range( 3 ):
        
        #The file could contain a {#} specification, like spectrum.pha{3},
        #which indicate the 3rd spectrum in the spectrum.pha file
        
        inputFiles[i] = fileUtils.sanitizeFilename( inputFiles[i].split("{")[0] )
        
        if not fileUtils.fileExistingAndReadable( inputFiles[i] ):
            
            raise IOError("File %s does not exist or is not readable" % ( inputFiles[i] ) )
    
    phafile, bkgfile, rspfile = inputFiles
    
    #Check the arf, if provided
    if arffile is not None:
        
        arffile = fileUtils.sanitizeFilename( arffile.split("{")[0] )
        
        if not fileUtils.fileExistingAndReadable( arffile ):
            
            raise IOError("File %s does not exist or is not readable" % ( arf ) )
    
    self.phafile              = OGIPPHA(phafile,filetype='observed')
    self.exposure             = self.phafile.getExposure()
    self.bkgfile              = OGIPPHA(bkgfile,filetype="background")
    self.response             = Response( rspfile, arffile )    
    
    #Start with an empty mask (the user will overwrite it using the
    #setActiveMeasurement method)
    self.mask                 = numpy.asarray(
                                    numpy.ones(self.phafile.getRates().shape),
                                    numpy.bool)
    
    #Get the counts for this spectrum
    self.counts               = ( self.phafile.getRates()[self.mask]
                                  * self.exposure )
    
    #Check that counts is positive
    idx                       = (self.counts < 0)
    
    if(numpy.sum(idx) > 0):
      
      warnings.warn("The observed spectrum for %s " % self.name + 
                    "has negative channels! Fixing those to zero.", 
                     RuntimeWarning)
      self.counts[idx]        = 0
    
    pass
    
    #Get the background counts for this spectrum
    self.bkgCounts            = ( self.bkgfile.getRates()[self.mask]
                                  * self.exposure )
    
    
    #Check that bkgCounts is positive
    idx                       = (self.bkgCounts < 0)
    
    if(numpy.sum(idx) > 0):
      
      warnings.warn("The background spectrum for %s " % self.name + 
                    "has negative channels! Fixing those to zero.", 
                     RuntimeWarning)
      self.bkgCounts[idx]     = 0
    
    #Check that the observed counts are positive
    
    idx = self.counts < 0
    
    if numpy.sum( idx ) > 0:
       
       raise RuntimeError("Negative counts in observed spectrum %s. Data are corrupted." % ( phafile ))
    
    #Keep a copy which will never be modified
    self.counts_backup        = numpy.array(self.counts,copy=True)
    self.bkgCounts_backup     = numpy.array(self.bkgCounts,copy=True)
    
    #Effective area correction is disabled by default, i.e.,
    #the nuisance parameter is fixed to 1    
    self.nuisanceParameters       = {}
    self.nuisanceParameters['InterCalib'] = Parameter("InterCalib",1,0.9,1.1,0.01,fixed=True,nuisance=True)
    
  pass
  
  def useIntercalibrationConst(self,factorLowBound=0.9,factorHiBound=1.1):
    self.nuisanceParameters['InterCalib'].free()
    self.nuisanceParameters['InterCalib'].setBounds(factorLowBound,factorHiBound)
    
    #Check that the parameter is within the provided bounds
    value                     = self.nuisanceParameters['InterCalib'].value
    
    if( value < factorLowBound ):
            
      warnings.warn("The intercalibration constant was %s, lower than the provided lower bound." %(value,factorLowBound) +
                    " Setting it equal to the lower bound")
      
      self.nuisanceParameters['InterCalib'].setValue(float(factorLowBound))

      
      
    if( value > factorHiBound):
      
      warnings.warn("The intercalibration constant was %s, larger than the provided hi bound." %(value,factorHiBound) +
                    " Setting it equal to the hi bound")
      
      self.nuisanceParameters['InterCalib'].setValue(float(factorHiBound))
    
  def fixIntercalibrationConst(self,value=None):
    
    if(value is not None):
      #Fixing the constant to the provided value
      self.nuisanceParameters['InterCalib'].setValue(float(value))
    
    else:
    
      #Do nothing, i.e., leave the constant to the value
      #it currently has
      pass
      
    self.nuisanceParameters['InterCalib'].fix()
  
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
    
    self.counts               = self.counts_backup[self.mask]
    self.bkgCounts            = self.bkgCounts_backup[self.mask]
    
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
        
    #Effective area correction
    if(self.nuisanceParameters['InterCalib'].isFree()):
            
      #A true fit would be an overkill, and slow
      #Just sample a 100 values and choose the minimum
      values                  = numpy.linspace(self.nuisanceParameters['InterCalib'].minValue,
                                               self.nuisanceParameters['InterCalib'].maxValue,
                                               100)
      
      
      #I do not use getLogLike so I can compute only once the folded model
      #(which is not going to change during the inner fit)
      
      folded                  = self.getFoldedModel()
      
      modelCounts             = folded * self.exposure
      
      def fitfun(cons):
        
        self.nuisanceParameters['InterCalib'].setValue( cons )
        
        return (-1) * self._computeLogLike(self.nuisanceParameters['InterCalib'].value * modelCounts + self.bkgCounts)
            
      logLval                 = map(fitfun, values)
      idx                     = numpy.argmax(logLval)
      self.nuisanceParameters['InterCalib'].setValue(values[idx])
      #return logLval[idx]
      
      #Now refine with minuit
      
      parameters              = collections.OrderedDict()
      parameters[ (self.name, 'InterCalib') ]      = self.nuisanceParameters['InterCalib']
      minimizer               = minimization.iMinuitMinimizer(fitfun, parameters)
      bestFit, mlogLmin       = minimizer.minimize()
      
      return mlogLmin * (-1)
      
    else:
      
      return self.getLogLike()
  
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
  
  def display( self ):
    
    #Plot the counts spectrum with residuals
    
    model, e1, e2, counts = self.getModelAndData()
    
    #Try to automagically decide a good constant number
    #of counts, based on the input
    
    total = numpy.sum( counts )
    
    trials = [ 50, 25, 15, 5]
    
    choice = None
    
    for t in trials:
        
        if total / float( t ) >= 10:
            
            choice = t
            
            break
    
    #If no choice worked, use the minimum, which is 5 counts
    #per bin
    
    if choice is None:
        
        choice = 5
    
    #Bin the data by constant counts
    
    binner = Binner.Binner( e1, e2, counts )
    
    ne1, ne2, nc, newModel = binner.byConstantCounts( choice, model )
        
    #Now plot the results
    
    fig = plt.figure()
    
    gs = gridspec.GridSpec(2,1, height_ratios=[2,1])
    gs.update(hspace=0)
    
    sub = plt.subplot( gs[0] )
    
    #Plot data
    
    ec = ( ne2 + ne1 ) / 2.0
    de = ( ne2 - ne1 ) / 2.0
    
    y = nc / ( de * 2.0 ) / self.exposure
    dy = numpy.sqrt( nc ) / ( de * 2.0 ) / self.exposure
    
    sub.errorbar( ec, y, xerr=de, yerr = dy, fmt='.', capsize=0 )
    
    #Plot model
    
    xx = numpy.append( ne1, [ ne2[-1] ])
    
    yy = newModel / ( 2.0 * de ) / self.exposure
    
    yyy = numpy.append( yy, yy[-1] )

    sub.step( xx, yyy, where='post' )
    
    sub.set_xscale("log")
    sub.set_yscale("log", nonposy='clip')
    
    sub.set_ylabel(r"Counts keV$^{-1}$ s$^{-1}$") 
    
    #Now plot residuals
    sub1 = plt.subplot( gs[1] )
    
    res = ( yy - y ) / dy
    
    sub1.axhline( 0, linestyle='--')
    sub1.errorbar( ec, res, yerr=1.0, xerr=de, fmt='.', capsize=0 )
    
    sub1.set_xscale("log")
    
    #Match the x axis
    
    sub.set_xticks([])
    
    sub.set_xlim( [( ec - de).min(), (ec + de).max() ] ) 
    sub1.set_xlim( [( ec - de).min(), (ec + de).max() ] ) 
    
    sub1.set_xlabel("Energy (keV)")
    
    return fig
  
  def _getModelCounts(self):
    
    #Get the folded model for this spectrum (this is the rate predicted, 
    #in cts/s)
    
    folded                    = self.getFoldedModel()
    
    #Model is folded+background (i.e., we assume negligible errors on the 
    #background)
    modelCounts               = self.nuisanceParameters['InterCalib'].value * folded * self.exposure + self.bkgCounts
    
    return modelCounts
  
  def _computeLogLike(self, modelCounts):
    
    idx = modelCounts > 0
    
    return numpy.sum(- modelCounts[idx] 
                     + self.counts[idx] * numpy.log( modelCounts[idx] )
                     - logfactorial( self.counts[idx] ) )
  
  def getLogLike(self):
    '''
    Return the value of the log-likelihood with the current values for the
    parameters
    '''
    
    modelCounts               = self._getModelCounts()
    
    logLike                   = self._computeLogLike( modelCounts )
    
    return logLike
      
  def getNuisanceParameters(self):
    '''
    Return a list of nuisance parameter names. Return an empty list if there
    are no nuisance parameters
    '''
    return self.nuisanceParameters.keys()
  pass
pass

class Response(object):
  def __init__( self, rspfile, arffile = None ):
    
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
      
      #Now let's see if we have a ARF, if yes, read it
      
      if arffile is not None:
          
          with pyfits.open( arffile ) as f:
              
              data = f['SPECRESP'].data
              
          
          arf = data.field('SPECRESP')
          
          #Check that arf and rmf have same dimensions
          
          if arf.shape[0] != self.matrix.shape[1]:
              
              raise IOError("The ARF and the RMF file does not have the same number of channels")
          
          #Check that the ENERG_LO and ENERG_HI for the RMF and the ARF
          #are the same
          
          arf_mc_channels = numpy.vstack([ data.field("ENERG_LO"),
                                           data.field("ENERG_HI") ]).T
          
          #Declare the mc channels different if they differ by more than
          #1%
          
          idx = ( self.mc_channels > 0 )
          
          diff = ( self.mc_channels[idx] - arf_mc_channels[idx] ) / self.mc_channels[idx]
          
          if diff.max() > 0.01:
             
             raise IOError("The ARF and the RMF have one or more MC channels which differ by more than 1%")
          
          #Multiply ARF and RMF
          
          self.matrix = self.matrix * arf
          
    
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
    
    #Sometimes some channels have 0 lenths, or maybe they start at 0, where
    #many functions (like a power law) are not defined. In the response these
    #channels have usually a 0, but unfortunately for a computer
    # inf * zero != zero. Thus, let's force this. We avoid checking this situation
    #in details because this would have a HUGE hit on performances
    
    idx = numpy.isfinite( trueFluxes )
    trueFluxes[ ~idx ] = 0    
    
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
