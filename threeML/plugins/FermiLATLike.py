import UnbinnedAnalysis
import BinnedAnalysis
import pyLikelihood as pyLike
import os
import shutil
import numpy
import pyfits
from threeML.pluginPrototype import pluginPrototype
from threeML.models.Parameter import Parameter

__instrument_name = "Fermi LAT (standard classes)"

class FermiLATLike(pluginPrototype):
  def __init__(self, name, ft2File, irf, livetimeCube, xmlModel, kind,*args):

    self.name                 = name
    
    self.ft2File              = ft2File   
    self.irf                  = irf
    self.livetimeCube         = livetimeCube
    
    #These are the boundaries and the number of energies for the computation
    #of the model
    self.emin                 = 1e4
    self.emax                 = 3e8
    self.Nenergies            = 1000
    
    #Make a copy of the xml model file and use that (it will be modified)
    self.xmlModel             = "__jointLikexml.xml"
    shutil.copyfile(xmlModel,self.xmlModel)
    
    #This is the limit on the effective area correction factor,
    #which is a multiplicative factor in front of the whole model
    #to account for inter-calibration issues. By default it can vary
    #by 10%. This can be changed by issuing:
    #FermiLATUnbinnedLikeInstance.effCorrLimit = [new limit]
    #where for example a [new limit] of 0.2 allow for an effective
    #area correction up to +/- 20 %
    
    self.effCorrLimit         = 0.1
    
    if(kind.upper()=='UNBINNED'):
      eventFile, exposureMap  = args
      self.eventFile          = eventFile
      self.exposureMap          = exposureMap
      #Read the files and generate the pyLikelihood object
      self.obs                  = UnbinnedAnalysis.UnbinnedObs(self.eventFile,
                                             self.ft2File,
                                             expMap=self.exposureMap,
                                             expCube=self.livetimeCube,
                                             irfs=self.irf)
    
      ##The following is quite slow (a couple of seconds)
      self.like                 = UnbinnedAnalysis.UnbinnedAnalysis(self.obs,
                                             self.xmlModel,
                                             optimizer='DRMNFB')
    elif(kind.upper()=="BINNED"):
       sourceMaps,binnedExpoMap  = args
       self.sourceMaps           = sourceMaps
       self.binnedExpoMap        = binnedExpoMap
       
       self.obs                  = BinnedAnalysis.BinnedObs(srcMaps=self.sourceMaps,
                                 	     expCube=self.livetimeCube,
                                 	     binnedExpMap=self.binnedExpoMap,
                                 	     irfs=self.irf)
       self.like                 = BinnedAnalysis.BinnedAnalysis(self.obs,
                                             self.xmlModel,
					     optimizer='DRMNFB')
    else:
      raise ValueError("FermiLATLike: 'kind' must be either BINNED or UNBINNED")
  pass
    
  def setModel(self,ModelManagerInstance):
    '''
    Set the model to be used in the joint minimization. Must be a ModelManager instance.
    '''
    self.modelManager         = ModelManagerInstance
    
    self._initFileFunction()
    
    #Here we need also to compute the logLike value, so that the model
    #in the XML file will be chanded if needed
    dumb                      = self.getLogLike()
    
    #Build the list of the nuisance parameters
    self._setNuisanceParameters()
  pass
  
  def getName(self):
    '''
    Return a name for this dataset (likely set during the constructor)
    '''
    return self.name
  pass
  
  def innerFit(self):
    '''
    This is used for the profile likelihood. Keeping fixed all parameters in the
    modelManager, this method minimize the logLike over the remaining nuisance
    parameters, i.e., the parameters belonging only to the model for this
    particular detector
    '''
    self._updateGtlikeModel()
    
    try:
      #Use .optimize instead of .fit because we don't need the errors
      #(.optimize is faster than .fit)
      self.like.optimize(0)
    except:
      #This is necessary because sometimes fitting algorithms go and explore extreme region of the
      #parameter space, which might turn out to give strange model shapes and therefore
      #problems in the likelihood fit
      print("Warning: failed likelihood fit (probably parameters are too extreme).")
      return 1e5
    else:
      #Update the value for the nuisance parameters
      for par in self.nuisanceParameters:
        newValue             = self.getNuisanceParameterValue(par.name)
        par.setValue(newValue)
      pass
      
      return self.like.logLike.value()  
  pass
  
  def _initFileFunction(self):
    #If the point source is already in the model, delete it
    #(like.deleteSource() returns the model)
    gtlikeSrcModel            = self.like.deleteSource(self.modelManager.name)
    
    #Get the new model for the source (with the latest parameter values)
    #and add it back to the likelihood model
    self._getNewGtlikeModel(gtlikeSrcModel,self.effCorrLimit)
    self.like.addSource(gtlikeSrcModel)
    
    #Slow! But no other options at the moment
    self.like.writeXml(self.xmlModel)
    self.like.logLike.reReadXml(self.xmlModel)
  pass
  
  def _getNewGtlikeModel(self,currentGtlikeModel,effAreaAllowedSize=0.1):
    #Write on disk the current model
    tempName                  = os.path.join(os.path.dirname(os.path.abspath(self.xmlModel)),'__fileSpectrum.txt')
    #This will recompute the model if necessary
    self.modelManager.writeToFile(tempName,self.emin,self.emax,self.Nenergies)
    
    #Generate a new FileFunction spectrum and assign it to the source
    fileFunction              = pyLike.FileFunction()
    fileFunction.readFunction(tempName)
    fileFunction.setParam("Normalization",1)
    p                         = fileFunction.parameter("Normalization")
    p.setBounds(1-float(effAreaAllowedSize),1+effAreaAllowedSize)
    
    currentGtlikeModel.setSpectrum(fileFunction)
  pass  
  
  def _updateGtlikeModel(self):    
    '''
    #Slow! But no other options at the moment
    self.like.writeXml(self.xmlModel)
    self.like.logLike.reReadXml(self.xmlModel)
    '''
    gtlikeSrcModel            = self.like[self.modelManager.name]
    self.modelManager.computeModel(self.emin,self.emax,self.Nenergies)
    
    my_function               = gtlikeSrcModel.getSrcFuncs()['Spectrum']
    my_file_function          = pyLike.FileFunction_cast(my_function)

    energies                  = self.modelManager.energies
    dnde                      = self.modelManager.values
    
    my_file_function.setParam("Normalization",1)
    my_file_function.setSpectrum(energies/1000.0, dnde*1000.0)
    gtlikeSrcModel.setSpectrum(my_file_function)
    #self.like.addSource(gtlikeSrcModel)
  pass
  
  def getLogLike(self):
    '''
    Return the value of the log-likelihood with the current values for the
    parameters stored in the ModelManager instance
    '''
    self._updateGtlikeModel()
    try:
      value                   = self.like.logLike.value()
    except:
      value                   = 1e5
    pass
    
    return value
  pass
  
  def getModelAndData(self):
    fake = numpy.array([])
    return fake,fake,fake,fake
  pass
  
  def _setNuisanceParameters(self):
    #Get the list of the sources
    sources                   = list(self.like.model.srcNames)
    
    freeParamNames            = []
    for srcName in sources:
      thisNamesV              = pyLike.StringVector()
      thisSrc                 = self.like.logLike.getSource(srcName)
      thisSrc.spectrum().getFreeParamNames(thisNamesV)
      thisNames               = map(lambda x:"%s-%s" %(srcName,x),thisNamesV)
      freeParamNames.extend(thisNames)
    pass
    
    self.nuisanceParameters   = []
    for name in freeParamNames:
      value                   = self.getNuisanceParameterValue(name)
      bounds                  = self.getNuisanceParameterBounds(name)
      delta                   = self.getNuisanceParameterDelta(name)
      self.nuisanceParameters.append(Parameter(name,value,bounds[0],bounds[1],delta,nuisance=True))
    pass
    
  pass
  
  def getNuisanceParameters(self):
    '''
    Return a list of nuisance parameters. Return an empty list if there
    are no nuisance parameters
    '''
    return self.nuisanceParameters
  
  
  def getNuisanceParameterValue(self,paramName):
    src,pname                 = paramName.split("-")
    return self.like.model[src].funcs['Spectrum'].getParam(pname).getValue()
  pass
  
  def getNuisanceParameterBounds(self,paramName):
    src,pname                 = paramName.split("-")
    return list(self.like.model[src].funcs['Spectrum'].getParam(pname).getBounds())
  pass
  
  def getNuisanceParameterDelta(self,paramName):
    src,pname                 = paramName.split("-")
    value                     = self.like.model[src].funcs['Spectrum'].getParam(pname).getValue()
    return value/100.0
  pass
  
  def setNuisanceParameterValue(self,paramName,value):
    src,pname                 = paramName.split("-")
    self.like.model[src].funcs['Spectrum'].getParam(pname).setValue(value)
  pass
pass
