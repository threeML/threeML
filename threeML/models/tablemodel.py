import astropy.io.fits as fits
import scipy.interpolate
import numpy
from threeML.models.spectralmodel import SpectralModel

class TableModel(SpectralModel):
    
    def SetTableFile(self,tableFile):
    
        self._ReadTable(tableFile)
        self._CreateInterpolation()
    
    def _ReadTable(self,tableFile):
        '''
        Virtual function. Implementation defined
        later.
        
        1) Should create an array called self._tableParams
        2) Create an array of evaluation energies
        3) Create a matrix of fluxes of shape (n_1,n_2,...n_i,j)
            where n are the number of values for the ith parameter
            and j is the number of evaluation energies
        
        '''
                
        pass
    
    def _CreateInterpolation(self):
        
        # Create the interpolation grid
        # 
        tmp = self._tableParams
        tmp.append(self._evalEnergies)
        
        interpGrid = tuple(tmp)
        
        
        self._interpFunc = scipy.interpolate.RegularGridInterpolator(interpGrid,self._tableFluxes,method="linear",fill_value=None)
        

        
        
class FITSTableModel(TableModel):
    
    def _ReadTable(self,fitsFileName):
        
        # Open the FITS file
        self._fitsFile = fits.open(fitsFileName)
        
        # Extract the evaluation energies
        self._ExtractEvalEnergies()
        
        # Extract the parameters
        self._ExtractParameters()
        
        # Extract the fluxes and reshape them
        # For the interpolator
        shape = self._numParamValues
        shape.append(self._numEvalEnergies)
        
        self._tableFluxes =  self._fitsFile[3].data['INTPSPEC'].reshape(*shape) / self._binWidths
        
        
        
        
    def _ExtractEvalEnergies(self):
    
        self._evalEnergies =  numpy.array(map(numpy.mean,self._fitsFile[2].data))
        self._binWidths = self._fitsFile[2].data["ENERG_HI"] - self._fitsFile[2].data["ENERG_LO"]
        
        self._numEvalEnergies = len(self._evalEnergies)
        
    def _ExtractParameters(self):
        
        self._numParamValues = (self._fitsFile[1].data["NUMBVALS"]).tolist()
        self._tableParams    = (self._fitsFile[1].data["VALUE"]).tolist()
    
    
