//Author: G.Vianello (giacomov@slac.stanford.edu)

//Interface class for the likelihood model
//Derived class must implement all these methods

#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H

#include <vector>
#include <string>

namespace threeML {
    
  class ModelInterface {
  
    public:
      
      //The use of "const" at the end of the method declaration promises
      //to the compiler that the method will not change the class
      //(in other words, these are all read-only methods)
            
      //Point source interface
      
      virtual int getNumberOfPointSources() const =0;
      
      virtual void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const =0;
      
      //Fluxes are differential fluxes in MeV^-1 cm^-1 s^-1
      virtual std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies) const =0;
      
      virtual std::string getPointSourceName(int srcid) const =0;
      
      //Extended source interface
      
      virtual int getNumberOfExtendedSources() const =0;
      
      virtual std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                   std::vector<double> energies) const =0;
      
      virtual std::string getExtendedSourceName(int srcid) const =0;
      
      virtual bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const =0;
      
      virtual void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                                  double *j2000_ra_max,
                                                  double *j2000_dec_min,
                                                  double *j2000_dec_max) const =0;
            
  };

}

#endif
