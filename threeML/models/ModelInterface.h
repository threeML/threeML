#include <utility>      // std::pair, std::make_pair
#include <map>
#include <vector>
#include <iostream>     // std::cout

#include <Python.h>
#include <boost/python.hpp>

namespace threeML {
  
  typedef std::vector<double> WrappableVector;
  
  class ModelInterface {
  
    public:
      
      ModelInterface(PyObject *pyModelUninterpreted);
      
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
      
      double m_nPtSources, m_nExtSources;
      
      typedef std::pair<double, double> skyPosition;
      
      std::map<int, skyPosition> m_ptsrcPos;
      
      boost::python::object m_pyModel;
      
  };

}
