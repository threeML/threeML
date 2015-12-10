//Author: G.Vianello (giacomov@slac.stanford.edu)

//This implements a real ModelInterface, which is used to bridge between a 3ML
//LikelihoodModel class (living in the python world) and plug-ins which are
//living in the C++ world. The plugins will only use the pyToCppModelInterface class
//to talk (without knowing it) to the python LikelihoodModel class.

#ifndef PYTOCPPMODEL_INTERFACE_H
#define PYTOCPPMODEL_INTERFACE_H

#include <boost/python.hpp>

#include <Python.h>

#include "ModelInterface.h"
#include <vector>
#include <map>
#include <string>

namespace threeML {
    
  class pyToCppModelInterface : public ModelInterface {
  
    public:
      
      pyToCppModelInterface(PyObject *pyModelUninterpreted);
      
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
                                                  double *j2000_dec_max)  const;
      
      void update();
      
    private:
      
      double m_nPtSources, m_nExtSources;
            
      boost::python::object m_pyModel;
      
      mutable int n_calls;
            
      mutable std::map<int, std::vector<double> > m_cache;
      
  };

}

#endif
