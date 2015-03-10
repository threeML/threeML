import astropy.io.fits as fits
import scipy.interpolate
import numpy
from threeML.models.spectralmodel import SpectralModel
import copy
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
        tmp = copy.deepcopy(self._tableParams)
        
        tmp.append(self._evalEnergies.tolist())
        tmp = map(lambda x: numpy.array(x,dtype=numpy.float32),tmp)
        interpGrid = tuple(tmp)
        self._tableFluxes.dtype = numpy.float32
        zero = numpy.float32(0.)
        
        self._interpFunc = scipy.interpolate.RegularGridInterpolator(interpGrid,self._tableFluxes,method="linear",fill_value=zero,bounds_error=False)
        

        
        
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
        
        self._tableFluxes =  self._fitsFile[3].data['INTPSPEC'].reshape(*shape) / self._binWidths #convert!
        
        
        
        
    def _ExtractEvalEnergies(self):
    
        self._evalEnergies =  numpy.array(map(numpy.mean,self._fitsFile[2].data))
        self._binWidths = self._fitsFile[2].data["ENERG_HI"] - self._fitsFile[2].data["ENERG_LO"]
        
        self._numEvalEnergies = len(self._evalEnergies)
        
    def _ExtractParameters(self):
        
        self._numParamValues = (self._fitsFile[1].data["NUMBVALS"]).tolist()
        self._tableParams    = (self._fitsFile[1].data["VALUE"]).tolist()
    
class NumpyTableModel(TableModel):

    def _ReadTable(self,npzfile):

        npz = numpy.load(npzfile)
        self._evalEnergies = npz["energy"]
        self._tableParams  = npz["params"].tolist()
        self._tableFluxes  = npz["fluxes"]


def MakeNumpyTableModel(params,evaluationEnergies,tableFluxes,filename):
    '''
    Helper function to save a table model into a numpy format.

    params:              a list of lists for the parameters values
    evalulationEnergies: array of the energies the model is evaluated at
    tablefluxes:         the table fluxes in a nested array
    filename: File name to save the table to. No extension required

    An example of how to build the table and params::

    f(ene,par1,par2) ---> the generating function
    
    param1 = [1,2,3,...] ---> the parameter values for par1
    param2 = [5,6,7,...] ---> the parameter values for par2

    eneGrid = [1,....,1e5] ---> the evaluation energies

    tabMod = []
    for p1 in par1:
     tmp = []
     for p2 in par2:
      tmp2.append(f(eneGrid,p1,p2))
     tmp.append(tmp2)
    tabMod.append(tmp)

    params = [param1,param2]


    MakeNumpyTable(params,eneGrid,tabMod,"table")    
    
    '''

    #Check the shape
    tableShape = tableFluxes.shape
    tmp = []
    for p in params:
        
        tmp.append(len(p))
    tmp.append(len(evaluationEnergies))
    
    if tuple(tmp) != tableShape:
        print "The parameters, evaluation energies and"
        print "table fluxes do not have the same shape!"
        print tuple(tmp)
        print tableShape
    
        return
    f = open(filename,"w")
    
    numpy.savez_compressed(f,params=params,energy=evaluationEnergies,fluxes=tableFluxes )
    
    f.close()
