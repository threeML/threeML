from band import Band
from scipy import weave
from scipy.weave import converters

class wBand(Band):
    def __call__(self,e):
      energies                 = numpy.array(e,ndmin=1,copy=False)
      alpha                    = self.parameters['alpha'].value
      beta                     = self.parameters['beta'].value
      E0                       = self.parameters['E0'].value
      K                        = self.parameters['K'].value
      
      n                        = energies.shape[0]
    
      code="""
        double *result = (double *)malloc(n*sizeof(double));
        int i;
        double ee,E,E100,ee100;
        ee = (alpha+beta)*E0;
        ee100 = ee/1000.0;
        for(i=0;i<n;i++)
        {
        E = energies(i);
        E100 = E/1000.0;
        if(E < ee) {
        result[i] = K*pow(E100,alpha)*exp(-E/E0);
        } else {
        result[i] = K*pow(ee100,alpha-beta)*exp(beta-alpha)*pow(E100,beta);
        }
        }
        return_val = result;
        """
      return weave.inline(code,['energies','alpha','beta','E0','K','n'],type_converters=converters.blitz)
  
 
