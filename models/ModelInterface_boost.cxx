#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "ModelInterface.h"
#include "FakePlugin.h"

using namespace boost::python;
using namespace threeML;


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


BOOST_PYTHON_MODULE(ModelInterface) 
{
  //hello
  to_python_converter<std::vector<double,std::allocator<double> >, VecToList<double> >();
      
  class_<ModelInterface>("ModelInterface",init<PyObject *>())
       .def("getNumberOfPointSources", &ModelInterface::getNumberOfPointSources)
       .def("getPointSourceFluxes", &ModelInterface::getPointSourceFluxes)
  ;
  
  class_<FakePlugin>("FakePlugin",init<ModelInterface>())
       .def("go", &FakePlugin::go)
       .def("createEnergies", &FakePlugin::createEnergies)
  ;
  
}
