import matplotlib.pyplot as plt
import numpy
import os
import Parameter
import collections
from threeML import fancyDisplay
from IPython.display import display, HTML

#The following color mapping has been taken from 
#http://colorbrewer2.org/?type=qualitative&scheme=Set1&n=9
#and has been selected for maximum contrast

colorPalette                  = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999',
                                 'red','blue','cyan','green','magenta','darkgrey']



class ModelManager(object):
  def __init__(self,name,spatialModel,spectralModel,dataList):
    '''
    Implements the model for a source
    '''
    
    #Default values:
    self.emin                 = 1 #keV
    self.emax                 = 1e9 #keV
    self.nbins                = 500
        
    self.name                 = name
    self.spatialModel         = spatialModel
    self.spectralModel        = spectralModel
    self.dataList             = dataList
        
    self.parameters           = collections.OrderedDict()
    for k,v in self.spatialModel.parameters.iteritems():
      self.parameters[k]      = v
      v.setCallback(self._scheduleRecomputing)
    for k,v in self.spectralModel.parameters.iteritems():
      self.parameters[k]      = v
      v.setCallback(self._scheduleRecomputing)
    
    self.modelToBeRecomputed  = True
    
    #Link the datasets to this modelManager
    for k,dataset in self.dataList.datasets.iteritems():
      dataset.setModel(self)
      thisNuisances           = dataset.getNuisanceParameters()
      for nuisance in thisNuisances:
        nuisance.setDataset(dataset)
        self.parameters[nuisance.name] = nuisance
      pass
    pass
  pass
    
  def getParameterNames(self):
    return self.parameters.keys()
  
  def getFreeParameters(self):
    freeParameters            = collections.OrderedDict()
    
    for k,v in self.parameters.iteritems():
      if(v.isFree()):
        freeParameters[k]     = v
      pass
    pass
    
    return freeParameters    
  
  def getFreeNormalizationParameters(self):
    normParameters            = collections.OrderedDict()
    
    for k,v in self.parameters.iteritems():
      if(v.isNormalization() and v.isFree()):
        normParameters[k]     = v
      pass
    pass
    
    return normParameters 
  
  def getFreeParameterNames(self):
    freeParameters            = []
    
    for k,v in self.parameters.iteritems():
      if(v.isFree()):
        freeParameters.append(v.name)
      pass
    pass
    
    return freeParameters
  pass
    
  def getNuisanceParameterNames(self):
    return self.nuisanceParameters.keys()
  pass
  
  def __getitem__(self,paramName):
    return self.parameters[paramName]
  
  def _scheduleRecomputing(self):
    #print("Scheduling")
    self.modelToBeRecomputed  = True
  
  def setNuisanceParameterValue(self,paramName,newValue):
    self.nuisanceParameters[paramName].dataset.setNuisanceParameterValue(paramName,newValue)
    self.nuisanceParameters[paramName].setValue(newValue)
  pass
  
  def setNuisanceParameterPrior(self,paramName,newPrior):
    self.nuisanceParameters[paramName].setPrior(newPrior)
  pass
    
  def computeModel(self,emin,emax,nbins):
    #Avoid recomputing if useless
    if(self.modelToBeRecomputed==False and self.emin==emin and self.emax==emax and self.nbins==nbins):
      return
    
    self.energies             = numpy.logspace(numpy.log10(emin),numpy.log10(emax),nbins)
    self.emin                 = emin
    self.emax                 = emax
    self.nbins                = nbins
    
    #To avoid numerical issue, I fix a floor value for the model to 1e-30
    self.values               = numpy.maximum(self.spectralModel(self.energies),1e-30)
    
    #Keep the model scheduled for recomputing if I am using a non-standard energy
    #bin
    self.modelToBeRecomputed  = False
  pass
    
  def plot(self,kind="counts",emin=1.0,emax=1e12,nbins=1000,**kwargs):
    #Compute the model with the current parameter values
    self.modelToBeRecomputed  = True
    self.computeModel(emin,emax,nbins)
    self.modelToBeRecomputed  = True
    sub                       = None
    
    #By default plot all datasets
    datasets                  = self.dataList.datasets.keys()
    for k,v in kwargs.iteritems():
      if(k.lower()=="subplot"):
        sub                   = v
      elif(k.lower()=="datasets"):
        try:
          datasets              = list(v)
        except:
          raise RuntimeError("You have to provide a list as argument for the 'datasets' keyword.")
      pass
    pass
    
    #plot it
    legend                    = False
    if(kind.lower()=="flux"):
      x                       = [self.energies]
      y                       = [self.values]
      xerr                    = [[]]
      yerr                    = [[]]
      labels                  = ['']
      alphas                  = [1]
      colors                  = [colorPalette[0]]
      xlabel                  = "keV"
      ylabel                  = r"Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$"
      styles                  = ['-']
    elif(kind.lower()=="nufnu"):
      x                       = [self.energies]
      keVtoErg                = 1.60217657e-9
      y                       = [self.values*numpy.power(numpy.array(self.energies),2)*keVtoErg]
      xerr                    = [[]]
      yerr                    = [[]]
      labels                  = ['']
      alphas                  = [1]
      colors                  = [colorPalette[0]]
      xlabel                  = "keV"
      ylabel                  = r"$\nu$ F$_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]"
      styles                  = ['-']
    elif(kind.lower()=="counts"):
      #Counts plot
      legend                  = True
      x                       = []
      xerr                    = []
      y                       = []
      yerr                    = []
      styles                  = []
      labels                  = []
      alphas                  = []
      colors                  = []
      for i,(k,dataset) in enumerate(self.dataList.datasets.iteritems()):
        if(not k in datasets):
          continue
        labels.append(k)
        folded,e1s,e2s,counts = dataset.getModelAndData()
        xmean                 = (e1s+e2s)/2.0
        de                    = (e2s-e1s)/2.0
        #Folded counts
        x.append(xmean)
        xerr.append(de)
        y.append(counts)
        yerr.append(numpy.sqrt(counts))
        styles.append(',')
        alphas.append(1)
        colors.append(colorPalette[i])
        #Model
        x.append(xmean)
        xerr.append([])
        y.append(folded)
        yerr.append([])
        styles.append('step')
        labels.append('')
        alphas.append(0.75)
        colors.append(colorPalette[i])
      pass
      
      xlabel                  = "keV"
      ylabel                  = r"Counts"
    else:
      raise RuntimeError("Unknown plot kind") 
    
    if(sub==None):
      fig,sub                   = plt.subplots(1,1)
    else:
      fig                       = sub.get_figure()
    pass
    
    for i in range(len(x)):
      args                    = {}
      if(legend and labels[i]!=''):
        args['label']         = labels[i]
      pass
      args['color']           = colors[i]
      args['alpha']           = alphas[i]
      
      if(len(xerr[i])==0 and len(yerr[i])==0):
        if(styles[i]=="step"):
          sub.step(x[i],y[i],where='mid',**args)
        else:
          sub.plot(x[i],y[i],styles[i],**args)
      else:
        sub.errorbar(x[i],y[i],xerr=xerr[i],yerr=yerr[i],fmt=styles[i],capsize=0,**args)
      pass
    pass
    
    #Optimize plot ranges
    
    
    sub.set_xscale("log",nonposx='clip')
    sub.set_yscale("log",nonposx='clip')
    sub.set_xlabel(xlabel)
    sub.set_ylabel(ylabel)
    sub.axis('tight')
    lims                      = plt.xlim()
    sub.set_xlim([lims[0]/2,lims[1]*2])
    lims                      = plt.ylim()
    sub.set_ylim([lims[0]/2,lims[1]*2])
    
    if(legend):
      sub.legend(loc=0,numpoints=1,ncol=2,fontsize='small')
    return fig
  pass
  
  def plotWithChain(self,parNames,samples,**kwargs):
    
    #Default values
    kind                      = 'flux'
    ymin                      = 1e-18
    emin                      = 1.0
    emax                      = 1e12
    nbins                     = 1000
    percentiles               = [16.0,50.0,84.0]
    subplot                   = None
    for k,v in kwargs.iteritems():
      if(k.lower()=="kind"):
        kind                  = v.lower()
      elif(k.lower()=="ymin"):
        ymin                  = float(v)
      elif(k.lower()=="emin"):
        emin                  = float(v)
      elif(k.lower()=="emax"):
        emax                  = float(v)
      elif(k.lower()=="nbins"):
        nbins                 = int(v)
      elif(k.lower()=="subplot"):
        subplot               = v
      elif(k.lower()=="percentiles"):
        percentiles           = list(v)
      elif(k.lower()=="energies"):
        #Use it later on 
        pass
      else:
        raise RuntimeError("Unknown keyword %s" %(k))
      pass
    pass
        
    #Compute the model with the current parameter values
    energies                  = numpy.logspace(numpy.log10(emin),numpy.log10(emax),nbins)
    if("energies" in map(lambda x:x.lower(),kwargs.keys())):
      energies                = kwargs["energies"]
    pass
    
    #Compute the quantiles
    #Compute the models for all the set of parameters
    
    modelValues               = numpy.zeros(shape=[energies.shape[0],len(samples)])
    for i,sample in enumerate(samples):
      map(lambda (key,val):self[key].setValue(val),
                                    zip(parNames,sample))      
      modelValues[:,i]        = self.spectralModel(energies)
    pass
    
    #Now for each energy get the quantiles
    median                    = numpy.zeros(energies.shape[0])
    lowQuantile               = numpy.zeros(energies.shape[0])
    hiQuantile                = numpy.zeros(energies.shape[0])
    for i in range(energies.shape[0]):
      l,m,h                   = numpy.percentile(modelValues[i,:],percentiles)
      median[i]               = m
      lowQuantile[i]          = l
      hiQuantile[i]           = h
    pass
    
    self.modelValues          = modelValues
    
    #Plot everything
    if(kind.lower()=="flux"):
      x                       = energies
      conv                    = 1.0
      xlabel                  = "keV"
      ylabel                  = r"Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$"
    elif(kind.lower()=="nufnu"):
      x                       = energies
      keVtoErg                = 1.60217657e-9
      conv                    = numpy.power(numpy.array(x)*keVtoErg,2)
      xlabel                  = "keV"
      ylabel                  = r"$\nu$ F$_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]"
    pass    
    
    if(subplot==None):
      fig,sub                   = plt.subplots(1,1)
    else:
      sub                       = subplot
      fig                       = sub.get_figure()
    pass
    
    sub.fill_between(energies,lowQuantile*conv,hiQuantile*conv,facecolor='green', alpha=0.4,linewidth=0.0)
    sub.plot(x,median*conv,'--')
    sub.loglog()
    sub.set_xlabel(xlabel)
    sub.set_ylabel(ylabel)
    sub.set_ylim([ymin,max(hiQuantile*conv)*1.1])
    
    self.median               = median
    self.lowQuantile          = lowQuantile
    self.hiQuantile           = hiQuantile
    return fig    
  pass
  
  def __repr__(self):
    print self.spatialModel
    print self.spectralModel
    
    toPrint                   = []
    for k,v in self.parameters.iteritems():
      if(v.isNuisance()):
        toPrint.append(v)
      pass
    pass
    
    print("\nNuisance parameters:\n")
    table                    = fancyDisplay.HtmlTable(7)
    table.addHeadings("Name","Value","Minimum","Maximum","Delta","Dataset","Unit")
    for v in toPrint:
      table.addRow(v.name,v.value,v.minValue,v.maxValue,v.delta,v.dataset.getName(),v.unit)
    pass
    display(HTML(table.__repr__()))

    return ''
  pass
  
  def printParamValues(self,output=False):
    string                    = []
    for k,v in self.parameters.iteritems():
      string.append("%s = %s" %(k,v.value))
    pass
    
    if(output):
      print(",".join(string))
    else:
      return ",".join(string)
  pass
  
  def writeToFile(self,filename,emin=None,emax=None,nbins=None):
    '''
    Write the spectral model to a file compatible with gtlike
    '''
    #Compute the model with the current parameter values
    self.computeModel(emin,emax,nbins)
    
    #Write it to file
    with open(filename,"w+") as f:
      modelString             = "#%s\n" % (self.spectralModel.functionName)
      for k,v in self.parameters.iteritems():
        modelString          += "#%s\n" % v.__repr__()
      pass
      for e,v in zip(self.energies,self.values):
        #Gtlike needs the energy in MeV and the flux in ph/cm2/s/MeV
        modelString          += "%.20g %.20g\n" %(e/1000.0,v*1000.0)
      f.write(modelString)
  pass  
pass
