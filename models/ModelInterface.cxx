#include "ModelInterface.h"
#include <boost/python/stl_iterator.hpp>

namespace threeML {

   template< typename T > inline
       std::vector< T > to_std_vector( const boost::python::object& iterable )
         {
           return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                                    boost::python::stl_input_iterator< T >( ) );
         }

  ModelInterface::ModelInterface(PyObject *pyModelUninterpreted) :
       m_nPtSources(0), m_nExtSources(0)
      { 
        std::cerr << "Building ModelInterface instance" << std::endl;
        
        boost::python::object 
            o(boost::python::handle<>(boost::python::borrowed(pyModelUninterpreted)));
        
        m_pyModel = o;
        
        //TODO: add a verification of the interface for the pyObject
        
        m_nPtSources = boost::python::extract<int>(m_pyModel.attr("nPtSources"));
        
      }
      
  bool ModelInterface::isInsideAnyExtendedSource(double lat, double lon) 
      { 
        return true;
      }
      
  int ModelInterface::getNumberOfPointSources() 
      {
        return m_nPtSources;
      }
      
  void ModelInterface::getPointSourcePosition(int srcid, double *lon, double *lat)
      {
        std::map<int, skyPosition>::iterator it = m_ptsrcPos.find(srcid);
        
        if(it == m_ptsrcPos.end()) {
          throw 10;
        }
        
        (*lon) = m_ptsrcPos[srcid].first;
        (*lat) = m_ptsrcPos[srcid].second;
      }
    
  std::vector<double> ModelInterface::getPointSourceFluxes(int srcid, std::vector<double> energies) 
      { 
                
        //Construct a generic object (instead of for example a list) so that
        //the pyModel can return any iterable (list, numpy.array, etc)
        boost::python::object fluxes = 
                        m_pyModel.attr("getPointSourceFluxes")(srcid,energies)
                        ;
        
        
        std::vector<double> fluxes_v = to_std_vector<double>(fluxes);
        return fluxes_v;
      }
      
  int ModelInterface::getNumberOfExtendedSources() 
      {
        return m_nExtSources;
      }
      
  std::vector<double> ModelInterface::getExtendedSourceFluxes(int srcid, double lon, double lat, 
                                   std::vector<double> energies)
      {
        std::vector<double> fluxes;
        return fluxes;
      }
  
  
  std::string ModelInterface::getPointSourceName(int srcid) 
      {
        std::string name("test");
        return name;
      }
  
  std::string getExtendedSourceName(int srcid)
      {
        std::string name("test");
        return name;
      }

}
