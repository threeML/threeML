import astropy.io.fits as pyfits
import xml.etree.ElementTree as ET
import os

from threeML.plugins.gammaln import logfactorial
from threeML.io import fileUtils
from threeML.plugins.GenericOGIPLike import GenericOGIPLike

import numpy
from threeML.plugins.ogip import OGIPPHA
from threeML.plugin_prototype import PluginPrototype
from threeML.models.Parameter import Parameter
from threeML.minimizer import minimization
import scipy.integrate

import warnings
import collections

__instrument_name = "Swift XRT"

class SwiftXRTLike( GenericOGIPLike ):
  
  def __init__(self, name, phafile, bkgfile, rspfile, arffile):
      
      super(SwiftXRTLike, self).__init__(name, phafile, bkgfile, rspfile, arffile)
