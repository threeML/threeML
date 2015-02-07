#include "pyToCppModelInterface.h"
#include <boost/python/stl_iterator.hpp>
#include <iostream>
#include <stdexcept>

namespace threeML {

   template< typename T > inline
       std::vector< T > to_std_vector( const boost::python::object& iterable )
         {
           return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                                    boost::python::stl_input_iterator< T >( ) );
         }

  pyToCppModelInterface::pyToCppModelInterface(PyObject *pyModelUninterpreted) :
       m_nPtSources(0), m_nExtSources(0)
      {         
        
        try 
          {
            boost::python::object 
                  o(boost::python::handle<>(boost::python::borrowed(pyModelUninterpreted)));
            
            m_pyModel = o;
            
          } 
        catch (...)
          {
            throw std::runtime_error("ModelInterface: Could not interpret the LikelihoodModel class");
          } 
        
        //TODO: add a verification of the interface for the pyObject
        
        try {
        
        
          m_nPtSources = boost::python::extract<int>(m_pyModel.attr("getNumberOfPointSources")());
        
        } catch (...) {
        
          throw std::runtime_error(
                      "ModelInterface: Could not use getNumberOfPointSources from python Object");
        
        }
      }
      
  bool pyToCppModelInterface::isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const
      { 
        return true;
      }
      
  int pyToCppModelInterface::getNumberOfPointSources() const
      {
        std::cerr << "NOOO" << std::endl;
        return m_nPtSources;
      }
      
  void pyToCppModelInterface::getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const
      {
        
        boost::python::object coords;
        
        try {
        
           coords = m_pyModel.attr("getPointSourcePosition")(srcid);
        
        } catch (...) {
        
          throw std::runtime_error(
                      "ModelInterface: Could not call getPointSourcePosition on the python side");
        
        }
        
        try {
        
          (*j2000_ra) = boost::python::extract<double>(coords[0]);
          (*j2000_dec) = boost::python::extract<double>(coords[1]);
          
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not convert the coordinates I got from the python side");
        }
      }
    
  std::vector<double> 
         pyToCppModelInterface::getPointSourceFluxes(int srcid, std::vector<double> energies) const
      { 
                
        //Construct a generic object (instead of for example a list) so that
        //the pyModel can return any iterable (list, numpy.array, etc)
        
        boost::python::object fluxes;
        
        try {
        
           fluxes = m_pyModel.attr("getPointSourceFluxes")(srcid,energies);
        
        } catch (...) {
          
          throw std::runtime_error(
                    "ModelInterface: Could not get the fluxes from the python side");
           
        }
        
        std::vector<double> fluxes_v;
        
        try {
          
           fluxes_v = to_std_vector<double>(fluxes);
        
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not convert the fluxes I got from the python side");
          
        }
        
        return fluxes_v;
      }
      
  int pyToCppModelInterface::getNumberOfExtendedSources() const
      {
        return 0;
      }
      
  std::vector<double> 
       pyToCppModelInterface::getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                                 std::vector<double> energies) const
      {
        std::vector<double> fluxes;
        return fluxes;
      }
  
  
  std::string pyToCppModelInterface::getPointSourceName(int srcid) const
      {
        
        std::string name;
        
        try {
        
           name = boost::python::extract<std::string>(
                                  m_pyModel.attr("getPointSourceName")(srcid)
                                  );
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not get the point source name from the python side");
        
        }
        
        return name;
      }
  
  std::string pyToCppModelInterface::getExtendedSourceName(int srcid) const
      {
        std::string name("test");
        return name;
      }

}
