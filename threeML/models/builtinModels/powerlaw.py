from threeML.models.spectralModels import SpectralModel



class PowerLaw(SpectralModel):
    def setup(self):
        self.functionName        = "Powerlaw"
        self.formula             = r'\begin{equation}f(E) = A E^{\gamma}\end{equation}'
        self.parameters          = collections.OrderedDict()
        self.parameters['gamma'] = Parameter('gamma',-2.0,-10,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['A']     = Parameter('A',1.0,1e-10,1e10,0.02,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['Epiv']  = Parameter('Epiv',1.0,1e-10,1e10,1,fixed=True)
    
        self.ncalls              = 0
    
    def integral(e1,e2):
        a                      = self.parameters['gamma'].value
        piv                    = self.parameters['Epiv'].value
        norm                   = self.parameters['A'].value
      
        if(a!=-1):
            def f(energy):
          
                return self.parameters['A'].value * math.pow(energy/piv,a+1)/(a+1)
        else:
            def f(energy):
                return self.parameters['A'].value * math.pow(energy/piv)
        return f(e2)-f(e1)
    self.integral            = integral
 
  
    def __call__(self,energy):
        self.ncalls             += 1
        piv                      = self.parameters['Epiv'].value
        norm                     = self.parameters['A'].value
        gamma                    = self.parameters['gamma'].value
        return numpy.maximum( numexpr.evaluate("norm * (energy/piv)**gamma"), 1e-30)
   
  
    def photonFlux(self,e1,e2):
        return self.integral(e1,e2)
  
    def energyFlux(self,e1,e2):
        a                        = self.parameters['gamma'].value
        piv                      = self.parameters['Epiv'].value
        if(a!=-2):
            def eF(e):
                return numpy.maximum(self.parameters['A'].value * numpy.power(e/piv,2-a)/(2-a),1e-30)
        else:
            def eF(e):
                return numpy.maximum(self.parameters['A'].value * numpy.log(e/piv),1e-30)
   
    
        return (eF(e2)-eF(e1))*keVtoErg

