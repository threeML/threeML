from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
import math
import scipy.integrate
import operator

import collections



class SmoothlyBrokenPowerlaw(SpectralModel):

    def setup(self):
        self.functionName        = "SmoothlyBrokenPowerlaw"
        self.formula             = r'\begin{equation}f(E) = A E^{\gamma}\end{equation}'

        self.parameters          = collections.OrderedDict()
        self.parameters['indx1'] = Parameter('indx1',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['indx2'] = Parameter('indx2',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['A']     = Parameter('A',1.0,1e-10,1e10,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['breakEnergy']  = Parameter('breakEnergy',300.,1.,1.e8,1,fixed=False)
        self.parameters['breakScale']  = Parameter('breakScale',1.,1.e-2,1.e1,.1,fixed=False)
        self.parameters['Epiv']  = Parameter('Epiv',300.,1.e0,1.e6,1.,fixed=True)
        
    
        self.ncalls              = 0
    
        #def integral(e1,e2):
        #    a                      = self.parameters['gamma'].value
        #    piv                    = self.parameters['Epiv'].value
        #    norm                   = self.parameters['A'].value
#      
#            if(a!=-1):
#                def f(energy):
#                    return self.parameters['A'].value * math.pow(energy/piv,a+1)/(a+1)
#            else:
#                def f(energy):
#                    return self.parameters['A'].value * math.pow(energy/piv)
#            return f(e2)-f(e1)
        self.integral            = None #Check this!
 
  
    def __call__(self,ene):
        self.ncalls             += 1
        pivot                      = self.parameters['Epiv'].value
        norm                     = self.parameters['A'].value
        indx1                    = self.parameters['indx1'].value
        indx2                    = self.parameters['indx2'].value
        breakE                   = self.parameters['breakEnergy'].value
        breakScale               = self.parameters['breakScale'].value

        val = numpy.zeros(ene.flatten().shape[0])
            
        

        B = (indx1 + indx2)/2.0
        M = (indx2 - indx1)/2.0

        arg_piv = numpy.log10(pivot/breakE)/breakScale

        if arg_piv < -6.0:

            pcosh_piv = M * breakScale * (-arg_piv-numpy.log(2.0))

        elif arg_piv > 4.0:

            pcosh_piv = M * breakScale * (arg_piv - numpy.log(2.0))

        else:

            pcosh_piv = M * breakScale * (numpy.log( (numpy.exp(arg_piv) + numpy.exp(-arg_piv))/2.0 ))



        arg = log10(ene/breakE)/breakScale


        idx1 =  arg < -6.0
        idx2 =  arg >  4.0
        idx3 =  ~numpy.logical_or(idx1,idx2)

        pcosh = numpy.zeros(ene.flatten().shape[0])

        pcosh[idx1] = M * breakScale * (-arg[idx1]-numpy.log(2.0))

        pcosh[idx2] = M * breakScale * (arg[idx2] - numpy.log(2.0))

        pcosh[idx3] = M * breakScale * (numpy.log( (numpy.exp(arg[idx3]) + numpy.exp(-arg[idx3]))/2.0 ))

        val = norm * numpy.power(ene/pivot,B)*numpy.power(10.,pcosh-pcosh_piv)

        return val

   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        pass
   
    
        #return (eF(e2)-eF(e1))*keVtoErg

