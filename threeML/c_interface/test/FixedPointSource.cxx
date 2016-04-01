#include "FixedPointSource.h"
#include <math.h>
#include <iostream>

namespace threeML {
  
  FixedPointSource::FixedPointSource(std::string name,
                                     double photonIndex, 
                                     double normalization,
                                     double pivotEnergy,
                                     double ra,
                                     double dec) :
       m_name(name),
       m_photonIndex(photonIndex),
       m_normalization(normalization),
       m_ra(ra),
       m_dec(dec),
       m_pivotEnergy(pivotEnergy),
       ModelInterface()
      { 
      
      }
  
  void FixedPointSource::describe() const
      {
        std::cout << "Fixed Point Source: " << std::endl;
        std::cout << "  Name:          " << m_name << std::endl;
        std::cout << "  R.A. (J2000):  " << m_ra << " deg" << std::endl;
        std::cout << "  Dec. (J2000):  " << m_dec << " deg" << std::endl;
        std::cout << "  Photon index:  " << m_photonIndex << std::endl;
        std::cout << "  Normalization: " << m_normalization << " (MeV^-1 cm^-2 s^-1)" << std::endl;
        std::cout << "  Pivot energy:  " << m_pivotEnergy << " (MeV)" << std::endl; 
      }
      
  bool FixedPointSource::isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const
      { 
        return false;
      }
      
  int FixedPointSource::getNumberOfPointSources() const
      {
                
        return 1;
      }
      
  void FixedPointSource::getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const
      {        
        (*j2000_ra) = m_ra;
        (*j2000_dec) = m_dec;
      }
    
  std::vector<double> FixedPointSource::getPointSourceFluxes(int srcid, std::vector<double> energies) const
      { 
        std::vector<double> fluxes(energies.size(),0.0);
        
        unsigned int n = energies.size();
        unsigned int i;
        
        for(i=0; i < n; ++i) 
        {
          fluxes[i] = m_normalization * pow(energies[i]/m_pivotEnergy,m_photonIndex);
        }
        
        return fluxes;
      }
      
  int FixedPointSource::getNumberOfExtendedSources() const
      {
        return 0;
      }
      
  std::vector<double> 
       FixedPointSource::getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                                 std::vector<double> energies) const
      {
        std::vector<double> fluxes(energies.size(),0.0);
        return fluxes;
      }
  
  
  std::string FixedPointSource::getPointSourceName(int srcid) const
      {
        return m_name;
      }
  
  std::string FixedPointSource::getExtendedSourceName(int srcid) const
      {
        std::string name("test");
        return name;
      }

}
