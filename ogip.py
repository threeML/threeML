#Implements a minimal reader for OGIP PHA format for spectral data
# (https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node6.html)

#Author: Giacomo Vianello (giacomov@slac.stanford.edu)

import os
import pyfits
import numpy

requiredKeywords             = {}
requiredKeywords['observed'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER,"+
                                 "exposure:EXPOSURE,backfile:BACKFILE,"+
                                 "corrfile:CORRFILE,corrscal:CORRSCAL,respfile:RESPFILE,"+
                                 "ancrfile:ANCRFILE,hduclass:HDUCLASS,"+
                                 "hduclas1:HDUCLAS1,hduvers:HDUVERS,poisserr:POISSERR,"+
                                 "chantype:CHANTYPE,n_channels:DETCHANS").split(",")
requiredKeywords['background'] = ("mission:TELESCOP,instrument:INSTRUME,filter:FILTER,"+
                                 "exposure:EXPOSURE,"+
                                 "hduclass:HDUCLASS,"+
                                 "hduclas1:HDUCLAS1,hduvers:HDUVERS,poisserr:POISSERR,"+
                                 "chantype:CHANTYPE,n_channels:DETCHANS").split(",")
mightBeColumns                = {}
mightBeColumns['observed']    = ("EXPOSURE,BACKFILE,"+
                                 "CORRFILE,CORRSCAL,"+
                                 "RESPFILE,ANCRFILE").split(",")
mightBeColumns['background']  = ("EXPOSURE").split(",")


class OGIPPHA(object):
  def __init__(self,phafile,spectrumNumber=None,**kwargs):
    
    self.filetype           = 'observed'
    for k,v in kwargs.iteritems():
      if(k.lower()=="filetype"):
        if(v.lower()=="background"):
          self.filetype       = "background"
        elif(v.lower()=="observed"):
          self.filetype       = "observed"
        else:
          raise RuntimeError("Unrecognized filetype keyword value")
      pass
    pass
    
    #Allow the use of a syntax like "mySpectrum.pha{1}" to specify the spectrum
    #number in PHA II files
    ext                     = os.path.splitext(phafile)[-1]
    if('{' in ext):
      spectrumNumber        = int(ext.split('{')[-1].replace('}',''))
      phafile               = phafile.split('{')[0]
    
    with pyfits.open(phafile) as f:
      try:
        HDUidx                = f.index_of("SPECTRUM")
      except:
        raise RuntimeError("The input file %s is not in PHA format" %(phafile))
      pass
      
      self.spectrumNumber     = spectrumNumber
      
      spectrum                = f[HDUidx]
      data                    = spectrum.data
      header                  = spectrum.header
      
      #Determine if this file contains COUNTS or RATES
      if("COUNTS" in data.columns.names):
        self.hasRates         = False
        self.dataColumnName   = "COUNTS"
      elif("RATE" in data.columns.names):
        self.hasRates         = True
        self.dataColumnName   = "RATE"
      else:
        raise RuntimeError("This file does not contain a RATE nor a COUNTS column. This is not a valid PHA file")
      pass
      
      #Determine if this is a PHA I or PHA II
      if(len(data.field(self.dataColumnName).shape)==2):
        self.typeII           = True
        if(self.spectrumNumber==None):
          raise RuntimeError("This is a PHA Type II file. You have to provide a spectrum number")
      else:
        self.typeII           = False
      pass
      
      #Collect informations from mandatory keywords
      keys                    = requiredKeywords[self.filetype]
      for k in keys:
        internalName,keyname  = k.split(":")
        if(keyname in header):
          self.__setattr__(internalName,header[keyname])
        else:
          if(keyname in mightBeColumns[self.filetype] and self.typeII):
            #Check if there is a column with this name
            if(keyname in data.columns.names):
              self.__setattr__(internalName,data[keyname][self.spectrumNumber-1])
            else:
              raise RuntimeError("Keyword %s is not in the header nor in the data extension. This file is not a proper PHA file" % keyname)
          else:
            #The keyword POISSERR is a special case, because even if it is missing,
            #it is assumed to be False if there is a STAT_ERR column in the file
            if(keyname=="POISSERR" and "STAT_ERR" in data.columns.names):
              self.poisserr   = False
            else:
              raise RuntimeError("Keyword %s not found. File %s is not a proper PHA file" % (keyname,phafile))
            pass
          pass
        pass
      pass
      
      #Now get the data (counts or rates) and their errors. If counts, transform them in rates
      if(self.typeII):
      
        #PHA II file
        if(self.hasRates):
          self.rates          = data.field(self.dataColumnName)[self.spectrumNumber-1,:]
          if(not self.poisserr):
            self.ratesErrors  = data.field("STAT_ERR")[self.spectrumNumber-1,:]
          pass
        else:
          self.rates          = data.field(self.dataColumnName)[self.spectrumNumber-1,:]/self.exposure
          if(not self.poisserr):
            self.ratesErrors  = data.field("STAT_ERR")[self.spectrumNumber-1,:]/self.exposure
        pass
        
        if("SYS_ERR" in data.columns.names):
            self.sysErrors    = data.field("SYS_ERR")[self.spectrumNumber-1,:]
        else:
            self.sysErrors    = numpy.zeros(self.rates.shape)
        pass
      
      elif(self.typeII==False):
        
        #PHA 1 file
        if(self.hasRates):
          self.rates          = data.field(self.dataColumnName)
          if(not self.poisserr):
            self.ratesErrors  = data.field("STAT_ERR")
        else:
          self.rates          = data.field(self.dataColumnName)/self.exposure
          if(not self.poisserr):
            self.ratesErrors  = data.field("STAT_ERR")/self.exposure
        pass
        
        if("SYS_ERR" in data.columns.names):
            self.sysErrors    = data.field("SYS_ERR")
        else:
            self.sysErrors    = numpy.zeros(self.rates.shape)
        pass
      pass
      
    pass #Closing the FITS file 
  pass
  
  def getRates(self):
    return self.rates
  
  def getRatesErrors(self):
    return self.ratesErrors
  
  def getSysErrors(self):
    return self.sysErrors
  
  def getExposure(self):
    return self.exposure
  
pass
