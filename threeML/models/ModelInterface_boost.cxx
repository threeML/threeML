#include <boost/python.hpp>

#include "ModelInterface.h"
#include "pyToCppModelInterface.h"
#include "FakePlugin.h"
#include "FixedPointSource.h"

using namespace boost::python;
using namespace threeML;

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
  ;
  
  class_<FakePlugin>("FakePlugin",init<ModelInterface *>())
       .def("go", &FakePlugin::go)
       .def("createEnergies", &FakePlugin::createEnergies)
  ;
  
  class_<FixedPointSource, bases<ModelInterface> >("FixedPointSource", 
                                                   init<std::string,
                                                        double,
                                                        double,
                                                        double,
                                                        double,
                                                        double>())
       .def("getNumberOfPointSources", &FixedPointSource::getNumberOfPointSources)
       .def("getPointSourceFluxes", &FixedPointSource::getPointSourceFluxes)
  ;
  
}
