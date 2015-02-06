#include "ModelInterface.h"
#include <string>

namespace threeML {
  
  class FixedPointSource : public ModelInterface 
  {
    public:
      
      //Predefined values are for the Crab at TeV energies
      
      FixedPointSource(std::string name = "Crab",
                       double photonIndex = -2.63, 
                       double normalization = 3.45e-5, //ph. MeV^-1 cm^-2 s^-1
                       double pivotEnergy = 1e6,
                       double ra = 83.63,
                       double dec = 22.01);
      
      void describe();
      
      //Point source interface
      
      int getNumberOfPointSources();
      
      void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec);
      
      //Fluxes are differential fluxes in MeV^-1 cm^-1 s^-1
      std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies);
      
      std::string getPointSourceName(int srcid);
      
      //Extended source interface
      
      int getNumberOfExtendedSources();
      
      std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                   std::vector<double> energies);
      
      std::string getExtendedSourceName(int srcid);
      
      bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec);
      
      void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                                  double *j2000_ra_max,
                                                  double *j2000_dec_min,
                                                  double *j2000_dec_max) {}
      

    
    private:
      
      std::string m_name;
      double m_photonIndex;
      double m_normalization;
      double m_ra;
      double m_dec;
      double m_pivotEnergy;
    
    
  };

}
