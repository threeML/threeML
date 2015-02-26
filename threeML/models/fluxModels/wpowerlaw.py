from powerlaw import PowerLaw
from scipy import weave
from scipy.weave import converters

class wPowerLaw(PowerLaw):
    def __call__(self,energy):
        a                         = self.parameters['A'].value
        alpha                     = self.parameters['gamma'].value
        try:
            n                       = energy.shape[0]
        except:
            n                       = 1
    
        result                    = numpy.array([0]*n,dtype=float)
        code="""
        int i;
        for(i=0;i<n;i++)
        {
        result(i) = a*pow(energy(i),alpha);
        }
        """
        weave.inline(code,['a','alpha','n','energy','result'],type_converters=converters.blitz)
        return result


