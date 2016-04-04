#include <boost/python.hpp>
#include "ModelInterface.h"
#include "pyToCppModelInterface.h"

#include <boost/python/stl_iterator.hpp>
#include <iostream>
#include <stdexcept>

//#include <boost/python.hpp>


namespace threeML {

   template< typename T > inline
       std::vector< T > to_std_vector( const boost::python::object& iterable )
         {
           return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                                    boost::python::stl_input_iterator< T >( ) );
         }

  pyToCppModelInterface::pyToCppModelInterface(PyObject *pyModelUninterpreted) :
       m_nPtSources(0), m_nExtSources(0), n_calls(0)
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
        
        try {
        
        
          m_nExtSources = boost::python::extract<int>(m_pyModel.attr("getNumberOfExtendedSources")());
        
        } catch (...) {
        
          throw std::runtime_error(
                      "ModelInterface: Could not use getNumberOfExtendedSources from python Object");
        
        }
        
      }
      
  bool pyToCppModelInterface::isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const
      { 
        return true;
      }
      
  int pyToCppModelInterface::getNumberOfPointSources() const
      {

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
  
  void pyToCppModelInterface::update()
      { 
         
         //std::cerr << "Emptying cache" << std::endl;
         
         //Empty the cache
         m_cache.clear();
      
      }
  
  std::vector<double> 
         pyToCppModelInterface::getPointSourceFluxes(int srcid, std::vector<double> energies) const
      {         
        
        if ( m_cache.count( srcid )==1 ) 
        {
          
          //std::cerr << "Cached" << std::endl;
          
          //Return cached version
        
          return m_cache[srcid];
        
        } else {
          
          n_calls += 1;
          
          //std::cerr << "Filling cache for " << srcid << " (" << n_calls << ")" << std::endl;
          
        }
        
        
        //Construct a generic object (instead of for example a list) so that
        //the pyModel can return any iterable (list, numpy.array, etc)
        
        boost::python::object fluxes;
        
        //try {
   

        //Transform MeV to keV
        
        for(unsigned int i=0; i < energies.size(); ++i) 
	      {
	       
	        energies[i] = energies[i] * 1000.0;
	       
	      }
	   
     
        //expects and returns MeV-related-units
        fluxes = m_pyModel.attr("getPointSourceFluxes")(srcid,energies);
        
        //} catch (...) {
          
        //  throw std::runtime_error(
        //            "ModelInterface: Could not get the fluxes from the python side");
           
        //}
        
        std::vector<double> fluxes_v;
        
        try {
          
           fluxes_v = to_std_vector<double>(fluxes);
	   
	   
           //Transform in ph/cm2/s/MeV from ph/cm2/s/keV
           
	         for(unsigned int i=0; i < fluxes_v.size(); ++i) 
	         {
	       
	           fluxes_v[i] = fluxes_v[i] * 1000.0;
	       
	         }
	   
        
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not convert the fluxes I got from the python side");
          
        }
        
        //Cache result
        
        m_cache[srcid] = fluxes_v;
        
        return fluxes_v;
      }
      
  int pyToCppModelInterface::getNumberOfExtendedSources() const
      {
        return m_nExtSources;
      }
      
  std::vector<double> 
       pyToCppModelInterface::getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                                 std::vector<double> energies) const
      {         
        
/*
        if ( m_cache.count( srcid * 1000 )==1 ) 
        {
          
          //std::cerr << "Cached" << std::endl;
          
          //Return cached version
        
          return m_cache[srcid * 1000];
        
        } else {
*/          
          n_calls += 1;
          
          //std::cerr << "Filling cache for " << srcid << " (" << n_calls << ")" << std::endl;
          
//        }
        
        
        //Construct a generic object (instead of for example a list) so that
        //the pyModel can return any iterable (list, numpy.array, etc)
        
        boost::python::object fluxes;
        
        //try {
   

        //Transform MeV to keV
        
        for(unsigned int i=0; i < energies.size(); ++i) 
	      {
	       
	        energies[i] = energies[i] * 1000.0;
	       
	      }
	   
     
        //expects and returns MeV-related-units
        fluxes = m_pyModel.attr("getExtendedSourceFluxes")(srcid, j2000_ra, j2000_dec, energies);
        
        //} catch (...) {
          
        //  throw std::runtime_error(
        //            "ModelInterface: Could not get the fluxes from the python side");
           
        //}
        
        std::vector<double> fluxes_v;
        
        try {
          
           fluxes_v = to_std_vector<double>(fluxes);
	   
	   
           //Transform in ph/cm2/s/MeV from ph/cm2/s/keV
           
	         for(unsigned int i=0; i < fluxes_v.size(); ++i) 
	         {
	       
	           fluxes_v[i] = fluxes_v[i] * 1000.0;
	       
	         }
	   
        
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not convert the fluxes I got from the python side");
          
        }
        
        //Cache result
        
//        m_cache[srcid * 1000] = fluxes_v;
        
        return fluxes_v;
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
        
        std::string name;
        
        try {
        
           name = boost::python::extract<std::string>(
                                  m_pyModel.attr("getExtendedSourceName")(srcid)
                                  );
        } catch (...) {
        
          throw std::runtime_error(
                    "ModelInterface: Could not get the extended source name from the python side");
        
        }
        
        return name;
      }

  void pyToCppModelInterface::getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                                          double *j2000_ra_max,
                                                          double *j2000_dec_min,
                                                          double *j2000_dec_max) const
    {
    
      boost::python::object boundaries;
      
      boundaries = m_pyModel.attr("getExtendedSourceBoundaries")( srcid );
      
      std::vector<double> boundaries_v = to_std_vector<double>(boundaries);
      
      *j2000_ra_min = boundaries_v[0];
      *j2000_ra_max = boundaries_v[1];
      *j2000_dec_min = boundaries_v[2];
      *j2000_dec_max = boundaries_v[3];
    
    }


}

using namespace threeML;
using namespace boost::python;

//This is needed to wrap the interface (i.e., all methods are virtual)
//contained in ModelInterface.h
struct ModelInterfaceWrap : ModelInterface, wrapper<ModelInterface>
{
    int getNumberOfPointSources() const { return this->get_override("getNumberOfPointSources")(); }
    
    void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const 
                                        { this->get_override("getPointSourcePosition")(); }
    
    std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies) const 
                                        { return this->get_override("getPointSourceFluxes")(); }
    
    std::string getPointSourceName(int srcid) const 
                                        { return this->get_override("getPointSourceName")(); }
    
    int getNumberOfExtendedSources() const 
                                        { return this->get_override("getNumberOfExtendedSources")(); }
    
    std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec, 
                                   std::vector<double> energies) const
                                        { return this->get_override("getExtendedSourceFluxes")(); }
    
    std::string getExtendedSourceName(int srcid) const 
                                        { return this->get_override("getExtendedSourceName")(); }
    
    bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const
                                        { return this->get_override("isInsideAnyExtendedSource")(); }
    
    void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                                  double *j2000_ra_max,
                                                  double *j2000_dec_min,
                                                  double *j2000_dec_max) const
                                        { this->get_override("getExtendedSourceBoundaries")(); }
};

template<class T>
struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        boost::python::list* l = new boost::python::list();
        
        for(size_t i = 0; i < vec.size(); i++)
            (*l).append(vec[i]);

        return l->ptr();
    }
};


BOOST_PYTHON_MODULE(pyModelInterface) 
{
  //hello
  to_python_converter<std::vector<double,std::allocator<double> >, VecToList<double> >();
  
  class_<ModelInterfaceWrap, boost::noncopyable>("ModelInterface")
    .def("getNumberOfPointSources", pure_virtual(&ModelInterface::getNumberOfPointSources))
    .def("getPointSourcePosition", pure_virtual(&ModelInterface::getPointSourcePosition))
    .def("getPointSourceFluxes", pure_virtual(&ModelInterface::getPointSourceFluxes))
    .def("getPointSourceName", pure_virtual(&ModelInterface::getPointSourceName))
    .def("getNumberOfExtendedSources", pure_virtual(&ModelInterface::getNumberOfExtendedSources))
    .def("getExtendedSourceFluxes", pure_virtual(&ModelInterface::getExtendedSourceFluxes))
    .def("getExtendedSourceName", pure_virtual(&ModelInterface::getExtendedSourceName))
    .def("isInsideAnyExtendedSource", pure_virtual(&ModelInterface::isInsideAnyExtendedSource))
    .def("getExtendedSourceBoundaries", pure_virtual(&ModelInterface::getExtendedSourceBoundaries))
    ;
  
  class_<pyToCppModelInterface, bases<ModelInterface> >("pyToCppModelInterface",init<PyObject *>())
       .def("getNumberOfPointSources", &pyToCppModelInterface::getNumberOfPointSources)
       .def("getPointSourceFluxes", &pyToCppModelInterface::getPointSourceFluxes)
       .def("update", &pyToCppModelInterface::update)
  ;
}
