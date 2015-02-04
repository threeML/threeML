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
      
      void getPointSourcePosition(int srcid, double *lon, double *lat);
      
      std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies);
      
      std::string getPointSourceName(int srcid);
      
      //Extended source interface
      
      int getNumberOfExtendedSources();
      
      std::vector<double> getExtendedSourceFluxes(int srcid, double lon, double lat, 
                                   std::vector<double> energies);
      
      std::string getExtendedSourceName(int srcid);
      
      bool isInsideAnyExtendedSource(double lat, double lon);
      
    private:
      
      double m_nPtSources, m_nExtSources;
      
      typedef std::pair<double, double> skyPosition;
      
      std::map<int, skyPosition> m_ptsrcPos;
      
      boost::python::object m_pyModel;
      
  };

}
