#ifndef FIXED_POINT_SOURCE_H
#define FIXED_POINT_SOURCE_H

#include <string>
#include <vector>
#include "ModelInterface.h"


namespace threeML {
      
  class FixedPointSource : public ModelInterface 
  {
    public:
      
      //Predefined values are for the Crab at TeV energies
      
      FixedPointSource(std::string = "Crab",
                       double photonIndex = -2.63, 
                       double normalization = 3.45e-17, //ph. MeV^-1 cm^-2 s^-1
                       double pivotEnergy = 1e6,
                       double ra = 83.63,
                       double dec = 22.01);
      
      //The use of "const" at the end of a method declaration means
      //that the method will not change anything in the class
      //(not even private members)
      
      void describe() const;
      
      //Point source interface
      
      int getNumberOfPointSources() const;
      
      void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const;
      
      //Fluxes are differential fluxes in MeV^-1 cm^-1 s^-1
      std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies) const;
      
      std::string getPointSourceName(int srcid) const;
      
      //Extended source interface
      
      int getNumberOfExtendedSources() const;
      
      std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                   std::vector<double> energies) const;
      
      std::string getExtendedSourceName(int srcid) const;
      
      bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const;
      
      void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                                  double *j2000_ra_max,
                                                  double *j2000_dec_min,
                                                  double *j2000_dec_max) const {}
      

    
    private:
      
      std::string m_name;
      double m_photonIndex;
      double m_normalization;
      double m_ra;
      double m_dec;
      double m_pivotEnergy;
    
    
  };

}

#endif
