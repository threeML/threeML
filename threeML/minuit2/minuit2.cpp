// minuit2.cpp is Copyright (C) 2008 Johann Cohen-Tanugi <cohen@slac.stanford.edu>
// See http://seal.web.cern.ch/seal/MathLibs/5_0_8/Minuit2/html/
// for more information about SEAL Minuit 1.7.9.
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
// 
// Full licence is in the file COPYING and at http://www.gnu.org/copyleft/gpl.html

#include "minuit2.h"
#include <iostream>

using namespace ROOT::Minuit2;
/*
 * PyVarObject_HEAD_INIT was added in Python 2.6.  Its use is
 * necessary to handle both Python 2 and 3.  This replacement
 * definition is for Python <=2.5
 */
#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size) \
    PyObject_HEAD_INIT(type) size,
#endif

#ifndef Py_TYPE
#define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#if PY_MAJOR_VERSION >= 3
#define MOD_DEF(ob, name, doc, methods) \
    static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
ob = PyModule_Create(&moduledef);
#else
#define MOD_DEF(ob, name, doc, methods) \
    ob = Py_InitModule3(name, methods, doc);
#endif

/*
 * Python 3 only has long.
 */
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#endif

#if PY_MAJOR_VERSION >= 3
#define PyString_Check(x) 1
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_FromFormat(x,y) PyUnicode_FromFormat(x,y)
#define PyString_AsString(x) PyUnicode_AS_DATA(x)
#endif

static PyObject *PyExc_MinuitError;

static PyMemberDef minuit_Minuit_members[] = {
   {"maxcalls", T_OBJECT, offsetof(minuit_Minuit, maxcalls), 0, "The maximum number of function calls before giving up on minimization."},
   {"tol", T_DOUBLE, offsetof(minuit_Minuit, tol), 0, "Tolerance: minimization succeeds when the estimated vertical distance to the\nminimum is less than 0.001*tol*up."},
   {"strategy", T_INT, offsetof(minuit_Minuit, strategy), 0, "Minimization strategy: 0 is fast, 1 is default, and 2 is thorough."},
   {"up", T_DOUBLE, offsetof(minuit_Minuit, up), 0, "The vertical distance from the minimum that corresponds to one standard\ndeviation.  This is 1.0 for chi^2 and 0.5 for -log likelihood."},
   {"printMode", T_INT, offsetof(minuit_Minuit, printMode), 0, "Call-by-call printouts: 0 shows nothing, 1 shows parameter values, 2 shows\ndifferences from the starting point, and 3 shows differences from the previous\nvalue."},
   {"fixed", T_OBJECT, offsetof(minuit_Minuit, fixed), 0, "Dictionary of fixed parameters; maps parameter strings to True/False."},
   {"limits", T_OBJECT, offsetof(minuit_Minuit, limits), 0, "Dictionary of domain limits; maps parameter strings to (low, high) or None for\nunconstrained fitting."},
   {"values", T_OBJECT, offsetof(minuit_Minuit, values), 0, "Dictionary of parameter values or starting points."},
    {"args", T_OBJECT, offsetof(minuit_Minuit, args), READONLY, "Tuple of parameters or starting points in the order of the objective function's argument list."},
   {"errors", T_OBJECT, offsetof(minuit_Minuit, errors), 0, "Dictionary of parameter errors or starting step sizes."},
   {"merrors", T_OBJECT, offsetof(minuit_Minuit, merrors), READONLY, "Dictionary of all MINOS errors that have been calculated so far."},
   {"covariance", T_OBJECT, offsetof(minuit_Minuit, covariance), READONLY, "Covariance matrix as a dictionary; maps pairs of parameter strings to matrix\nelements."},

   {"fcn", T_OBJECT, offsetof(minuit_Minuit, fcn), READONLY, "The objective function: must accept only numeric arguments and return a number."},
   {"fval", T_OBJECT, offsetof(minuit_Minuit, fval), READONLY, "The current minimum value of the objective function."},
   {"ncalls", T_INT, offsetof(minuit_Minuit, ncalls), READONLY, "The number of times the objective function has been called: also known as NFCN."},
   {"edm", T_OBJECT, offsetof(minuit_Minuit, edm), READONLY, "The current estimated vertical distance to the minimum."},
   {"parameters", T_OBJECT, offsetof(minuit_Minuit, parameters), READONLY, "A tuple of parameters, in the order of the objective function's argument list."},

   {NULL}
};

static PyMethodDef minuit_Minuit_methods[] = {
    {"migrad", (PyCFunction)(minuit_Minuit_migrad), METH_NOARGS, "Attempt to minimize the function with the MIGRAD algorithm (recommended).  No output if successful: see member values\nand errors."},
    {"simplex", (PyCFunction)(minuit_Minuit_simplex), METH_NOARGS, "Attempt to minimize the function with the Simplex algorithm (not recommended).  No output if successful: see member values\nand errors."},
   {"hesse", (PyCFunction)(minuit_Minuit_hesse), METH_NOARGS, "Measure the covariance matrix with the current values."},
   {"minos", (PyCFunction)(minuit_Minuit_minos), METH_VARARGS, "Measure non-linear error bounds after minimization.  For all parameters, pass\nno arguments; for one parameter, pass parameter name and number of sigmas\n(negative for the other side)."},
   {"contour", (PyCFunction)(minuit_Minuit_contour), METH_VARARGS | METH_KEYWORDS, "Measure a 2-dimensional contour line, given two parameter strings, a number of\nsigmas, and (optionally) a number of points."},
   {"scan", (PyCFunction)(minuit_Minuit_scan), METH_VARARGS | METH_KEYWORDS, "Crudely minimize the function by scanning in N dimensions.  Arguments are\n(parameter, bins, low, high), ..., for all parameters of interest.  Keyword\narguments corners=True measures left edges, rather than centers of bins and\noutput=False suppresses the output tensor of measured values."},
   {"matrix", (PyCFunction)(minuit_Minuit_matrix), METH_VARARGS | METH_KEYWORDS, "Express the covariance as a tuple-of-tuples matrix.  Optional correlation=True\ncalculates the (normalized) correlation matrix instead, and skip_fixed=True\nremoves fixed parameters (which have zeroed entries)."},
   {NULL}
};

static PyTypeObject minuit_MinuitType = {
    PyVarObject_HEAD_INIT(NULL,0)
        "minuit2.Minuit2",      /*tp_name*/
   sizeof(minuit_Minuit), /*tp_basicsize*/
   0,                         /*tp_itemsize*/
   (destructor)minuit_Minuit_dealloc,    /*tp_dealloc*/
   0,                         /*tp_print*/
   0,                         /*tp_getattr*/
   0,                         /*tp_setattr*/
   0,                         /*tp_compare*/
   0,                         /*tp_repr*/
   0,                         /*tp_as_number*/
   0,                         /*tp_as_sequence*/
   0,                         /*tp_as_mapping*/
   0,                         /*tp_hash */
   0,                         /*tp_call*/
   0,                         /*tp_str*/
   0,                         /*tp_getattro*/
   0,                         /*tp_setattro*/
   0,                         /*tp_as_buffer*/
   Py_TPFLAGS_DEFAULT,        /*tp_flags*/
   "Represents a function to be minimized by Minuit.  Pass a Python callable, and\noptionally param1=2., param2=3., ... to set initial values, err_param1=0.5 to\nset initial step sizes, fix_param1=True to prevent parameters from floating in\nthe fit, and limit_param1=(low, high) to set limits.\n\nTo minimize, call minuit(), for a covariance matrix, call hesse(), for\nnon-linear errors, minos() or minos(param, nsigmas), and for 2-dimensional\ncontour lines, call contour(param1, param2, nsigmas).  You can also scan the\nfunction with scan((param1, bins, low, high), (param2, bins, low, high), ...).", /* tp_doc */
   0,		               /* tp_traverse */
   0,		               /* tp_clear */
   0,		               /* tp_richcompare */
   0,		               /* tp_weaklistoffset */
   0,		               /* tp_iter */
   0,		               /* tp_iternext */
   minuit_Minuit_methods, /* tp_methods */
   minuit_Minuit_members, /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)minuit_Minuit_init, /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};

static int minuit_Minuit_init(minuit_Minuit *self, PyObject *args, PyObject *kwds) {
   self->myfcn = NULL;
   self->upar = NULL;
   self->min = NULL;
   self->scandepth = 0;
   self->maxcalls = NULL;
   self->fixed = NULL;
   self->limits = NULL;
   self->values = NULL;
   self->args = NULL;
   self->errors = NULL;
   self->merrors = NULL;
   self->covariance = NULL;
   self->fval = NULL;
   self->edm = NULL;
   self->self = NULL;

   PyObject *arg = NULL;
   if (!PyArg_ParseTuple(args, "O", &arg)  ||  !PyCallable_Check(arg)) {
       PyErr_SetString(PyExc_TypeError, "First argument must be a callable function, instance method or instance.");
       return -1;
   }

   PyObject *function = NULL;
    if (PyFunction_Check(arg)){
        function = arg;
    }
    else if (PyMethod_Check(arg)){
        function = PyMethod_Function(arg);
        self->self = PyMethod_Self(arg);
        if (!self->self){
            PyErr_SetString(PyExc_TypeError, "Unbound methods are not supported.");
      return -1;
   }
        Py_INCREF(self->self);
    }
    else {
        // __call__ has to exist because we checked that the object is callable
        arg = PyObject_GetAttrString(arg,"__call__");
        function = PyMethod_Function(arg);
        self->self = PyMethod_Self(arg);
        if (!self->self){
            PyErr_SetString(PyExc_TypeError, "Unbound methods are not supported.");
            return -1;
        }
        Py_DECREF(arg);
        Py_INCREF(self->self);
    }

   self->fcn = function;
   Py_INCREF(self->fcn);

    PyObject *func_code = PyFunction_GetCode(self->fcn);
   if (func_code == NULL) {
      return -1;
   }
   PyObject *co_varnames = PyObject_GetAttrString(func_code, "co_varnames");
   if (co_varnames == NULL) {
      return -1;
   }
   PyObject *co_argcount = PyObject_GetAttrString(func_code, "co_argcount");
   if (co_argcount == NULL) {
      Py_DECREF(co_varnames);
      return -1;
   }
   if (!PyTuple_Check(co_varnames)) {
      PyErr_SetString(PyExc_TypeError, "function.func_code.co_varnames must be a tuple.");
      Py_DECREF(co_varnames);
      Py_DECREF(co_argcount);
      return -1;
   }
   if (!PyInt_Check(co_argcount)) {
      PyErr_SetString(PyExc_TypeError, "function.func_code.co_argcount must be an integer.");
      Py_DECREF(co_varnames);
      Py_DECREF(co_argcount);
      return -1;
   }
   if (PyInt_AsLong(co_argcount) < 1) {
      PyErr_SetString(PyExc_TypeError, "This function has no parameters to minimize.");
      Py_DECREF(co_varnames);
      Py_DECREF(co_argcount);
      return -1;
   }

    // ensure that self->parameters does not contain self method parameter
    if (self->self){
        self->parameters = PyTuple_GetSlice(co_varnames, 1, PyInt_AsLong(co_argcount));
    }
    else{
   self->parameters = PyTuple_GetSlice(co_varnames, 0, PyInt_AsLong(co_argcount));
    }

   Py_DECREF(co_varnames);
   Py_DECREF(co_argcount);

   self->npar = PyTuple_Size(self->parameters);
   self->maxcalls = Py_BuildValue("O", Py_None);
   self->tol = 0.1;
   self->strategy = 1;
   self->up = 1.;
   self->printMode = 0;
   self->fixed = PyDict_New();
   self->limits = PyDict_New();
   self->values = PyDict_New();
   self->args = Py_BuildValue("O", Py_None);
   self->errors = PyDict_New();
   self->merrors = PyDict_New();
   self->covariance = Py_BuildValue("O", Py_None);
   self->fval = Py_BuildValue("O", Py_None);
   self->ncalls = 0;
   self->edm = Py_BuildValue("O", Py_None);

   self->upar = new MnUserParameters();

    Py_DECREF(self->args);
    self->args = PyTuple_New(self->npar);

    for (int i = 0;  i < self->npar;  i++) {
      PyObject *param = PyTuple_GetItem(self->parameters, i);
      if (!PyString_Check(param)) {
	 PyErr_SetString(PyExc_RuntimeError, "function.func_code.co_varnames must be a tuple of strings.");
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
	 return -1;
      }

      double value = 0.;
      double error = 0.1;

      if (kwds != 0  &&  PyDict_Contains(kwds, param) == 1) {
	 if (!PyNumber_Check(PyDict_GetItem(kwds, param))) {
	    PyErr_SetString(PyExc_TypeError, "All values must be numbers.");
         Py_DECREF(self->args);
         self->args = Py_BuildValue("O", Py_None);
	    return -1;
	 }
	 value = PyFloat_AsDouble(PyDict_GetItem(kwds, param));
      }

      PyObject *pyvalue = Py_BuildValue("d", value);
      if (PyDict_SetItem(self->values, param, pyvalue) != 0) {
	 Py_DECREF(pyvalue);
      Py_DECREF(self->args);
      self->args = Py_BuildValue("O", Py_None);
	 return -1;
      }

        if (PyTuple_SetItem(self->args, i, pyvalue) != 0) {
      Py_DECREF(pyvalue);
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
            return -1;
        }

      PyObject *err_param = PyString_FromFormat("err_%s", PyString_AsString(param));
      if (kwds != 0  &&  PyDict_Contains(kwds, err_param) == 1) {
	 if (!PyNumber_Check(PyDict_GetItem(kwds, err_param))) {
	    PyErr_SetString(PyExc_TypeError, "All errors must be numbers.");
                Py_DECREF(self->args);
                self->args = Py_BuildValue("O", Py_None);
	    return -1;
	 }
	 error = PyFloat_AsDouble(PyDict_GetItem(kwds, err_param));
      }

      PyObject *pyerror = Py_BuildValue("d", error);
      if (PyDict_SetItem(self->errors, param, pyerror) != 0) {
	 Py_DECREF(err_param);
	 Py_DECREF(pyerror);
	 return -1;
      }
      Py_DECREF(pyerror);

      Py_DECREF(err_param);

      PyObject *fixvalue = Py_False;
      PyObject *fix_param = PyString_FromFormat("fix_%s", PyString_AsString(param));
      if (kwds != 0  &&  PyDict_Contains(kwds, fix_param) == 1) {
	 fixvalue = PyDict_GetItem(kwds, fix_param);
	 if (fixvalue != Py_True  &&  fixvalue != Py_False) {
	    PyErr_Format(PyExc_TypeError, "fix_%s must be True or False.", PyString_AsString(param));
                Py_DECREF(self->args);
                self->args = Py_BuildValue("O", Py_None);
	    return -1;
	 }
      }

      if (PyDict_SetItem(self->fixed, param, fixvalue) != 0) {
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
	 return -1;
      }

      PyObject *limitvalue = Py_None;
      PyObject *limit_param = PyString_FromFormat("limit_%s", PyString_AsString(param));
      if (kwds != 0  &&  PyDict_Contains(kwds, limit_param) == 1) {
	 limitvalue = PyDict_GetItem(kwds, limit_param);
	 if (limitvalue != Py_None) {
	    if (!(PyTuple_Check(limitvalue)  &&  PyTuple_Size(limitvalue) == 2)) {
	       PyErr_Format(PyExc_TypeError, "limit_%s must be None or (low, high).", PyString_AsString(param));
                    Py_DECREF(self->args);
                    self->args = Py_BuildValue("O", Py_None);
	       return -1;
	    }
	    if (!PyNumber_Check(PyTuple_GetItem(limitvalue, 0))) {
	       PyErr_Format(PyExc_TypeError, "limit_%s[0] (lower limit) must be a number.", PyString_AsString(param));
                    Py_DECREF(self->args);
                    self->args = Py_BuildValue("O", Py_None);
	       return -1;
	    }
	    if (!PyNumber_Check(PyTuple_GetItem(limitvalue, 1))) {
	       PyErr_Format(PyExc_TypeError, "limit_%s[1] (upper limit) must be a number.", PyString_AsString(param));
                    Py_DECREF(self->args);
                    self->args = Py_BuildValue("O", Py_None);
	       return -1;
	    }
	 }
      }

      if (PyDict_SetItem(self->limits, param, limitvalue) != 0) {
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
	 return -1;
      }

        Py_DECREF(limit_param);
        Py_DECREF(fix_param);

      self->upar->Add(PyString_AsString(param), value, error);
   }
   
    self->myfcn = new MyFCN(self->fcn, self->self, self->npar);
    self->myfcn->SetUp(self->up);
    self->myfcn->SetPrintMode(self->printMode);

   return 0;
}

static int minuit_Minuit_dealloc(minuit_Minuit *self) {

    delete self->myfcn;
    delete self->upar;
    delete self->min;

   self->myfcn = NULL;
   self->upar = NULL;
   self->min = NULL;

    Py_XDECREF(self->self);
    Py_XDECREF(self->fcn);
    Py_XDECREF(self->parameters);
    Py_XDECREF(self->maxcalls);
    Py_XDECREF(self->fixed);
    Py_XDECREF(self->limits);
    Py_XDECREF(self->values);
    Py_XDECREF(self->args);
    Py_XDECREF(self->errors);
    Py_XDECREF(self->merrors);
    Py_XDECREF(self->covariance);
    Py_XDECREF(self->fval);
    Py_XDECREF(self->edm);

    Py_TYPE(self)->tp_free((PyObject*)self);
   return 0;
}

bool minuit_prepare(minuit_Minuit *self, int &maxcalls, std::vector<std::string> &floating) {
   maxcalls = 0;

   if (self->maxcalls == Py_None) { /* 0 means no limit */ }
   else if (PyInt_Check(self->maxcalls)) {
      maxcalls = int(PyInt_AsLong(self->maxcalls));
      if (maxcalls <= 0) {
	 PyErr_SetString(PyExc_ValueError, "maxcalls must be positive (set to None for no limit).");
	 return false;
      }
   }
   else {
      PyErr_SetString(PyExc_TypeError, "maxcalls must be an integer or None");
      return false;
   }

   if (self->tol <= 0.) {
      PyErr_SetString(PyExc_ValueError, "tol must be positive.");
      return false;
   }

   if (self->strategy != 0  &&  self->strategy != 1  &&  self->strategy != 2) {
      PyErr_SetString(PyExc_ValueError, "strategy must be 0, 1, or 2.");
      return false;
   }

   if (self->up <= 0.) {
      PyErr_SetString(PyExc_ValueError, "tol must be positive.");
      return false;
   }

   if (self->printMode != 0  &&  self->printMode != 1  &&  self->printMode != 2  &&  self->printMode != 3) {
      PyErr_SetString(PyExc_ValueError, "printMode must be 0, 1, 2, or 3.");
      return false;
   }

   if (!PyDict_Check(self->values)) {
      PyErr_SetString(PyExc_TypeError, "values must be a dictionary.");
      return false;
   }

   if (!PyDict_Check(self->errors)) {
      PyErr_SetString(PyExc_TypeError, "errors must be a dictionary.");
      return false;
   }

   if (!PyDict_Check(self->fixed)) {
      PyErr_SetString(PyExc_TypeError, "fixed must be a dictionary.");
      return false;
   }

   if (!PyDict_Check(self->limits)) {
      PyErr_SetString(PyExc_TypeError, "limits must be a dictionary.");
      return false;
   }

   int nfixed = 0;
   floating.clear();
   for (int i = 0;  i < self->npar;  i++) {
      PyObject *value = PyDict_GetItemString(self->values, self->upar->Name(i));
      if (value == NULL) {
	 PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from values.", self->upar->Name(i));
	 return false;
      }
      if (!PyNumber_Check(value)) {
	 PyErr_Format(PyExc_TypeError, "values[\"%s\"] must be a number.", self->upar->Name(i));
	 return false;
      }
      self->upar->SetValue(i, PyFloat_AsDouble(value));

      PyObject *error = PyDict_GetItemString(self->errors, self->upar->Name(i));
      if (error == NULL) {
	 PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from errors.", self->upar->Name(i));
	 return false;
      }
      if (!PyNumber_Check(error)) {
	 PyErr_Format(PyExc_TypeError, "errors[\"%s\"] must be a number.", self->upar->Name(i));
	 return false;
      }
      self->upar->SetError(i, PyFloat_AsDouble(error));

      PyObject *fixed = PyDict_GetItemString(self->fixed, self->upar->Name(i));
      if (fixed == NULL) {
	 PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from fixed.", self->upar->Name(i));
	 return false;
      }
      if (fixed != Py_True  &&  fixed != Py_False) {
	 PyErr_Format(PyExc_TypeError, "fixed[\"%s\"] must be True or False.", self->upar->Name(i));
	 return false;
      }
      if (fixed == Py_True) {
	 if (!self->upar->Parameter(i).IsFixed()) {
	    self->upar->Fix(i);
	 }
	 nfixed++;
      }
      else {
	 if (self->upar->Parameter(i).IsFixed()) {
	    self->upar->Release(i);
	 }
	 floating.push_back(std::string(self->upar->Name(i)));
      }

      PyObject *limit = PyDict_GetItemString(self->limits, self->upar->Name(i));
      if (limit == NULL) {
	 PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from limits.", self->upar->Name(i));
	 return false;
      }
      if (limit == Py_None) {
	 self->upar->RemoveLimits(i);
      }
        else if (PySequence_Check(limit)  &&  PySequence_Size(limit) == 2) {
            if (!PyNumber_Check(PySequence_GetItem(limit, 0))) {
	    PyErr_Format(PyExc_TypeError, "limits[\"%s\"][0] (lower limit) must be a number.", self->upar->Name(i));
	    return false;
	 }
            if (!PyNumber_Check(PySequence_GetItem(limit, 1))) {
	    PyErr_Format(PyExc_TypeError, "limits[\"%s\"][1] (upper limit) must be a number.", self->upar->Name(i));
	    return false;
	 }

            if (PyFloat_AsDouble(PySequence_GetItem(limit, 0)) >= PyFloat_AsDouble(PySequence_GetItem(limit, 1))) {
	    PyErr_Format(PyExc_ValueError, "limits[\"%s\"] has lower limit >= upper limit.", self->upar->Name(i));
	    return false;
	 }

	 self->upar->RemoveLimits(i);
            self->upar->SetLimits(i, PyFloat_AsDouble(PySequence_GetItem(limit, 0)), PyFloat_AsDouble(PySequence_GetItem(limit, 1)));
      }
      else {
	 PyErr_Format(PyExc_TypeError, "limits[\"%s\"] must be None or (low, high).", self->upar->Name(i));
	 return false;
      }
   }

   if (nfixed >= self->npar) {
      PyErr_SetString(PyExc_RuntimeError, "Can't minimize if all parameters are fixed.");
      return false;
   }

    self->myfcn->SetUp(self->up);
    self->myfcn->SetPrintMode(self->printMode);
   if (self->printMode > 0) {
      switch (self->printMode) {
	 case 1:
	    printf("  FCN Result | Parameter values\n");
	    printf("-------------+--------------------------------------------------------\n");
	    break;
	 case 2:
	    printf("  FCN Result | Differences in parameter values from initial\n");
	    printf("-------------+--------------------------------------------------------\n");
                self->myfcn->SetOriginal(self->upar->Params());
	    break;
	 case 3:
	    printf("  FCN Result | Differences in parameter values from the previous\n");
	    printf("-------------+--------------------------------------------------------\n");
                self->myfcn->SetOriginal(self->upar->Params());
	    break;
      }
   }
   
   return true;
}

static PyObject *minuit_Minuit_simplex(minuit_Minuit *self) {
    int maxcalls = 0;
    std::vector<std::string> floating;
    if (!minuit_prepare(self, maxcalls, floating)) {
        return NULL;
    }

    MnSimplex simplex(*self->myfcn, *self->upar, self->strategy);

    if (self->min != NULL) {
        delete self->min;
        self->min = NULL;
    }

    try {
        self->min = new FunctionMinimum(simplex(maxcalls, self->tol));
    }
    catch (ExceptionDuringMinimization theException) {
        if (self->min != NULL) {
            delete self->min;
            self->min = NULL;
        }
        return NULL;
    }

    Py_DECREF(self->fval);
    self->fval = PyFloat_FromDouble(self->min->Fval());

    self->ncalls = self->min->NFcn();

    Py_DECREF(self->edm);
    self->edm = PyFloat_FromDouble(self->min->Edm());

    Py_DECREF(self->args);
    self->args = PyTuple_New(self->npar);

    for (int i = 0;  i < self->npar;  i++) {
        PyObject *value = PyFloat_FromDouble(self->min->UserParameters().Value(i));
        if (PyDict_SetItemString(self->values, self->upar->Name(i), value) != 0) {
            Py_DECREF(value);
            if (self->min != NULL) {
                delete self->min;
                self->min = NULL;
            }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
            return NULL;
        }

        if (PyTuple_SetItem(self->args, i, value) != 0) {
            Py_DECREF(value);
            if (self->min != NULL) {
                delete self->min;
                self->min = NULL;
            }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
            return NULL;
        }

        PyObject *error = PyFloat_FromDouble(self->min->UserParameters().Error(i));
        if (PyDict_SetItemString(self->errors, self->upar->Name(i), error) != 0) {
            Py_DECREF(error);
            if (self->min != NULL) {
                delete self->min;
                self->min = NULL;
            }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
            return NULL;
        }
        Py_DECREF(error);
    }

    if (self->min->HasValidCovariance()) {
        MnUserCovariance ucov(self->min->UserCovariance());
        PyObject *cov = PyDict_New();

        for (unsigned int i = 0;  i < floating.size();  i++) {
            for (unsigned int j = 0;  j < floating.size();  j++) {
                PyObject *key = Py_BuildValue("ss", floating[i].c_str(), floating[j].c_str());
                if (key == NULL) {
                    Py_DECREF(cov);
                    if (self->min != NULL) {
                        delete self->min;
                        self->min = NULL;
                    }
                    return NULL;
                }

                PyObject *val = PyFloat_FromDouble(ucov(i, j));
                if (val == NULL) {
                    Py_DECREF(cov);
                    Py_DECREF(key);
                    if (self->min != NULL) {
                        delete self->min;
                        self->min = NULL;
                    }
                    return NULL;
                }

                if (PyDict_SetItem(cov, key, val) != 0) {
                    Py_DECREF(cov);
                    if (self->min != NULL) {
                        delete self->min;
                        self->min = NULL;
                    }
                    return NULL;
                }

                Py_DECREF(key);
                Py_DECREF(val);
            }
        }

        Py_DECREF(self->covariance);
        self->covariance = cov;
        // one reference to cov == self->covariance is already counted
    }
    else {
        Py_DECREF(self->covariance);
        self->covariance = Py_BuildValue("O", Py_None);
    }

    if (!self->min->IsValid()) {
        if (self->min->HasReachedCallLimit()) {
            PyErr_SetString(PyExc_MinuitError, "Minuit reached the specified call limit (maxcalls).");
        }
        else if (self->min->IsAboveMaxEdm()) {
            PyErr_SetString(PyExc_MinuitError, "Function value is above the specified estimated distance to the minimum (edm).");
        }
        else if (!self->min->HasPosDefCovar()) {
            PyErr_SetString(PyExc_MinuitError, "Covariance is not positive definite.");
        }
        else if (!self->min->HasMadePosDefCovar()) {
            PyErr_SetString(PyExc_MinuitError, "Covariance could not be made positive definite.");
        }
        else if (!self->min->HasAccurateCovar()) {
            PyErr_SetString(PyExc_MinuitError, "Covariance is not accurate.");
        }
        else if (!self->min->HasValidCovariance()) {
            PyErr_SetString(PyExc_MinuitError, "Covariance is not valid.");
        }
        else if (self->min->HesseFailed()) {
            PyErr_SetString(PyExc_MinuitError, "HESSE failed.");
        }
        else if (!self->min->HasValidParameters()) {
            PyErr_SetString(PyExc_MinuitError, "Parameters are not valid.");
        }
        else {
            PyErr_SetString(PyExc_MinuitError, "Minuit failed.");
        }
        if (self->min != NULL) {
            delete self->min;
            self->min = NULL;
        }
        return NULL;
    }

    return Py_BuildValue("O", Py_None);
}

static PyObject *minuit_Minuit_migrad(minuit_Minuit *self) {
   int maxcalls = 0;
   std::vector<std::string> floating;
   if (!minuit_prepare(self, maxcalls, floating)) {
      return NULL;
   }

   MnMigrad migrad(*self->myfcn, *self->upar, self->strategy);

   if (self->min != NULL) {
      delete self->min;
      self->min = NULL;
   }

   try {
      self->min = new FunctionMinimum(migrad(maxcalls, self->tol));
   }
   catch (ExceptionDuringMinimization theException) {
      if (self->min != NULL) {
	 delete self->min;
	 self->min = NULL;
      }
      return NULL;
   }

   Py_DECREF(self->fval);
   self->fval = PyFloat_FromDouble(self->min->Fval());

   self->ncalls = self->min->NFcn();

   Py_DECREF(self->edm);
   self->edm = PyFloat_FromDouble(self->min->Edm());

    Py_DECREF(self->args);
    self->args = PyTuple_New(self->npar);

   for (int i = 0;  i < self->npar;  i++) {
      PyObject *value = PyFloat_FromDouble(self->min->UserParameters().Value(i));
      if (PyDict_SetItemString(self->values, self->upar->Name(i), value) != 0) {
	 Py_DECREF(value);
	 if (self->min != NULL) {
	    delete self->min;
	    self->min = NULL;
	 }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
	 return NULL;
      }

        if (PyTuple_SetItem(self->args, i, value) != 0) {
      Py_DECREF(value);
            if (self->min != NULL) {
                delete self->min;
                self->min = NULL;
            }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
            return NULL;
        }

      PyObject *error = PyFloat_FromDouble(self->min->UserParameters().Error(i));
      if (PyDict_SetItemString(self->errors, self->upar->Name(i), error) != 0) {
	 Py_DECREF(error);
	 if (self->min != NULL) {
	    delete self->min;
	    self->min = NULL;
	 }
            Py_DECREF(self->args);
            self->args = Py_BuildValue("O", Py_None);
	 return NULL;
      }
      Py_DECREF(error);
   }

   if (self->min->HasValidCovariance()) {
      MnUserCovariance ucov(self->min->UserCovariance());
      PyObject *cov = PyDict_New();

      for (unsigned int i = 0;  i < floating.size();  i++) {
	 for (unsigned int j = 0;  j < floating.size();  j++) {
	    PyObject *key = Py_BuildValue("ss", floating[i].c_str(), floating[j].c_str());
	    if (key == NULL) {
	       Py_DECREF(cov);
	       if (self->min != NULL) {
		  delete self->min;
		  self->min = NULL;
	       }
	       return NULL;
	    }

	    PyObject *val = PyFloat_FromDouble(ucov(i, j));
	    if (val == NULL) {
	       Py_DECREF(cov);
	       Py_DECREF(key);
	       if (self->min != NULL) {
		  delete self->min;
		  self->min = NULL;
	       }
	       return NULL;
	    }
	    
	    if (PyDict_SetItem(cov, key, val) != 0) {
	       Py_DECREF(cov);
	       if (self->min != NULL) {
		  delete self->min;
		  self->min = NULL;
	       }
	       return NULL;
	    }

	    Py_DECREF(key);
	    Py_DECREF(val);
	 }
      }

      Py_DECREF(self->covariance);
      self->covariance = cov;
      // one reference to cov == self->covariance is already counted
   }
   else {
      Py_DECREF(self->covariance);
      self->covariance = Py_BuildValue("O", Py_None);
   }

   if (!self->min->IsValid()) {
      if (self->min->HasReachedCallLimit()) {
	 PyErr_SetString(PyExc_MinuitError, "Minuit reached the specified call limit (maxcalls).");
      }
      else if (self->min->IsAboveMaxEdm()) {
	 PyErr_SetString(PyExc_MinuitError, "Function value is above the specified estimated distance to the minimum (edm).");
      }
      else if (!self->min->HasPosDefCovar()) {
	 PyErr_SetString(PyExc_MinuitError, "Covariance is not positive definite.");
      }
      else if (!self->min->HasMadePosDefCovar()) {
	 PyErr_SetString(PyExc_MinuitError, "Covariance could not be made positive definite.");
      }
      else if (!self->min->HasAccurateCovar()) {
	 PyErr_SetString(PyExc_MinuitError, "Covariance is not accurate.");
      }
      else if (!self->min->HasValidCovariance()) {
	 PyErr_SetString(PyExc_MinuitError, "Covariance is not valid.");
      }
      else if (self->min->HesseFailed()) {
	 PyErr_SetString(PyExc_MinuitError, "HESSE failed.");
      }
      else if (!self->min->HasValidParameters()) {
	 PyErr_SetString(PyExc_MinuitError, "Parameters are not valid.");
      }
      else {
	 PyErr_SetString(PyExc_MinuitError, "Minuit failed.");
      }
      if (self->min != NULL) {
	 delete self->min;
	 self->min = NULL;
      }
      return NULL;
   }

   return Py_BuildValue("O", Py_None);
}

static PyObject *minuit_Minuit_hesse(minuit_Minuit *self) {
   int maxcalls = 0;
   std::vector<std::string> floating;
   if (!minuit_prepare(self, maxcalls, floating)) {
      return NULL;
   }

   MnHesse hesse(self->strategy);
   MnUserParameterState ustate;

   try {
      ustate = hesse(*self->myfcn, *self->upar, maxcalls);
   }
   catch (ExceptionDuringMinimization theException) {
      return NULL;
   }

   self->ncalls = ustate.NFcn();

   for (int i = 0;  i < self->npar;  i++) {
      PyObject *error = PyFloat_FromDouble(ustate.Error(i));
      if (PyDict_SetItemString(self->errors, self->upar->Name(i), error) != 0) {
	 Py_DECREF(error);
	 return NULL;
      }
      Py_DECREF(error);
   }

   if (ustate.HasCovariance()) {
      MnUserCovariance ucov(ustate.Covariance());
      PyObject *cov = PyDict_New();

      for (unsigned int i = 0;  i < floating.size();  i++) {
	 for (unsigned int j = 0;  j < floating.size();  j++) {
	    PyObject *key = Py_BuildValue("ss", floating[i].c_str(), floating[j].c_str());
	    if (key == NULL) {
	       Py_DECREF(cov);
	       return NULL;
	    }

	    PyObject *val = PyFloat_FromDouble(ucov(i, j));
	    if (val == NULL) {
	       Py_DECREF(cov);
	       Py_DECREF(key);
	       return NULL;
	    }
	    
	    if (PyDict_SetItem(cov, key, val) != 0) {
	       Py_DECREF(cov);
	       return NULL;
	    }

	    Py_DECREF(key);
	    Py_DECREF(val);
	 }
      }

      Py_DECREF(self->covariance);
      self->covariance = cov;
      // one reference to cov == self->covariance is already counted
   }
   else {
      Py_DECREF(self->covariance);
      self->covariance = Py_BuildValue("O", Py_None);
   }

   if (!ustate.IsValid()) {
      PyErr_SetString(PyExc_MinuitError, "HESSE failed.");
      return NULL;
   }

   return Py_BuildValue("O", Py_None);
}

static PyObject *minuit_Minuit_minos(minuit_Minuit *self, PyObject *args) {
   if (args == NULL  ||  PyTuple_Size(args) == 0) {
      for (int i = 0;  i < self->npar;  i++) {
	 PyObject *subargs = Py_BuildValue("Od", PyTuple_GetItem(self->parameters, i), -1.);
	 if (minuit_Minuit_minos(self, subargs) == NULL) {
	    Py_DECREF(subargs);
	    return NULL;
	 }
	 Py_DECREF(subargs);

	 subargs = Py_BuildValue("Od", PyTuple_GetItem(self->parameters, i), 1.);
	 if (minuit_Minuit_minos(self, subargs) == NULL) {
	    Py_DECREF(subargs);
	    return NULL;
	 }
	 Py_DECREF(subargs);
      }

      return Py_BuildValue("O", Py_None);
   }

   char *param;
   double sigmas;
   if (!PyArg_ParseTuple(args, "sd", &param, &sigmas)) {
      PyErr_SetString(PyExc_TypeError, "Either pass no arguments or parameter, number of sigmas.");
      return NULL;
   }

   if (sigmas == 0.) {
      PyErr_SetString(PyExc_TypeError, "Number of sigmas may not be zero.");
      return NULL;
   }

   int index = -1;
   for (int i = 0;  i < self->npar;  i++) {
      if (strcmp(param, self->upar->Name(i)) == 0) {
	 index = i;
	 break;
      }
   }
   if (index == -1) {
      PyErr_Format(PyExc_TypeError, "\"%s\" is not a parameter.", param);
      return NULL;
   }

   if (self->min == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "You must run MIGRAD (successfully) first.");
      return NULL;
   }

   int maxcalls = 0;
   std::vector<std::string> floating;
   if (!minuit_prepare(self, maxcalls, floating)) {
      return NULL;
   }

    self->myfcn->SetUp(self->up * sigmas*sigmas);

   MnMinos minos(*self->myfcn, *self->min, self->strategy);
   MnCross crossing;
   try {
      if (sigmas > 0.) {
	 crossing = minos.Upval(index, maxcalls);
      }
      else {
	 crossing = minos.Loval(index, maxcalls);
      }
   }
   catch (ExceptionDuringMinimization theException) {
      return NULL;
   }
   
   self->ncalls = crossing.NFcn();

   if (crossing.NewMinimum()) {
      PyErr_Format(PyExc_MinuitError, "Discovered a new minimum at %s = %g", param, crossing.State().Value(index));
      return NULL;
   }

   if (!crossing.IsValid()) {
      if (crossing.AtLimit()) {
	 PyErr_Format(PyExc_MinuitError, "Reached the edge of the \"%s\" parameter's domain.", param);
	 return NULL;
      }
      else if (crossing.AtMaxFcn()) {
	 PyErr_SetString(PyExc_MinuitError, "Minuit reached the specified call limit (maxcalls).");
	 return NULL;
      }
      else {
	 PyErr_SetString(PyExc_MinuitError, "MINOS failed.");
	 return NULL;
      }
   }

   double minos_error;
   if (sigmas > 0.) {
      minos_error = crossing.State().Error(index) * (1. + crossing.Value());
   }
   else {
      minos_error = -1.*crossing.State().Error(index) * (1. + crossing.Value());
   }
   PyObject *key = Py_BuildValue("sd", param, sigmas);
   PyObject *val = PyFloat_FromDouble(minos_error);
   if (PyDict_SetItem(self->merrors, key, val) != 0) {
      Py_DECREF(key);
      Py_DECREF(val);
      return NULL;
   }
   Py_DECREF(key);
   Py_DECREF(val);

   return Py_BuildValue("O", Py_None);
}

static PyObject *minuit_Minuit_contour(minuit_Minuit *self, PyObject *args, PyObject *kwds) {
   char *param1, *param2;
   double sigmas;
   int npoints = 20;
    static char *kwlist[] = {"param1", "param2", "sigmas", "npoints", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssd|i", kwlist, &param1, &param2, &sigmas, &npoints)) {
      PyErr_SetString(PyExc_TypeError, "Arguments are: param1, param2, number of sigmas, and optionally approximate number of points (20).");
      return NULL;
   }

   if (sigmas <= 0.) {
      PyErr_SetString(PyExc_TypeError, "Number of sigmas must be positive.");
      return NULL;
   }

   if (npoints < 3) {
      PyErr_SetString(PyExc_TypeError, "Number of points must be at least 3.");
      return NULL;
   }

   int index1 = -1;
   for (int i = 0;  i < self->npar;  i++) {
      if (strcmp(param1, self->upar->Name(i)) == 0) {
	 index1 = i;
	 break;
      }
   }
   if (index1 == -1) {
      PyErr_Format(PyExc_TypeError, "\"%s\" is not a parameter.", param1);
      return NULL;
   }

   int index2 = -1;
   for (int i = 0;  i < self->npar;  i++) {
      if (strcmp(param2, self->upar->Name(i)) == 0) {
	 index2 = i;
	 break;
      }
   }
   if (index2 == -1) {
      PyErr_Format(PyExc_TypeError, "\"%s\" is not a parameter.", param2);
      return NULL;
   }

   if (index1 == index2) {
      PyErr_SetString(PyExc_ValueError, "The two parameters must be different.");
      return NULL;
   }

   if (self->min == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "You must run MIGRAD (successfully) first.");
      return NULL;
   }

   int maxcalls = 0;
   std::vector<std::string> floating;
   if (!minuit_prepare(self, maxcalls, floating)) {
      return NULL;
   }

    self->myfcn->SetUp(self->up * sigmas*sigmas);

   MnContours contours(*self->myfcn, *self->min, self->strategy);
   std::vector<std::pair<double, double> > points;

   try {
      ContoursError conterr = contours.Contour(index1, index2, npoints);
      points = conterr();
      self->ncalls = conterr.NFcn();
   }
   catch (ExceptionDuringMinimization theException) {
      return NULL;
   }
   
   PyObject *output = PyList_New(points.size());
   int i = 0;
   std::vector<std::pair<double, double> >::const_iterator pend = points.end();
   for (std::vector<std::pair<double, double> >::const_iterator p = points.begin();  p != pend;  ++p) {
      PyObject *item = Py_BuildValue("dd", p->first, p->second);
      if (PyList_SetItem(output, i, item) != 0) {
	 Py_DECREF(output);
	 Py_DECREF(item);
	 return NULL;
      }
      i++;			 
   }

   return output;
}

static PyObject* minuit_Minuit_scan(minuit_Minuit* self, PyObject* args, PyObject* kwds) {
   const char *errstring = "Arguments are: (\"param\", bins, low, high), ..., output=False, corners=False";
   if (args == NULL  ||  PyTuple_Size(args) == 0) {
      PyErr_SetString(PyExc_TypeError, errstring);
      self->scandepth = 0;
      return NULL;
   }

   PyObject *subargs = NULL;
   if (PyTuple_Size(args) > 1) {
      subargs = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
   }

   PyObject *specification = PyTuple_GetItem(args, 0);
   if (specification == NULL) {
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   if (PyTuple_Size(specification) != 4) {
      PyErr_SetString(PyExc_TypeError, errstring);
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   
   PyObject *param = PyTuple_GetItem(specification, 0);
   int bins = int(PyInt_AsLong(PyTuple_GetItem(specification, 1)));
   double low = PyFloat_AsDouble(PyTuple_GetItem(specification, 2));
   double high = PyFloat_AsDouble(PyTuple_GetItem(specification, 3));

   if (PySequence_Contains(self->parameters, param) != 1) {
        PyErr_SetString(PyExc_ValueError, "Unidentified parameter name.");
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   if (bins < 1) {
      PyErr_Format(PyExc_ValueError, "Parameter \"%s\" bins must be at least 1.", PyString_AsString(param));
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   if (low >= high) {
      PyErr_Format(PyExc_ValueError, "Parameter \"%s\" low must be less than high.", PyString_AsString(param));
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }

   if (self->printMode != 0  &&  self->printMode != 1  &&  self->printMode != 2  &&  self->printMode != 3) {
      PyErr_SetString(PyExc_ValueError, "printMode must be 0, 1, 2, or 3.");
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   if (self->printMode > 0) {
      switch (self->printMode) {
	 case 1:
	    printf("  FCN Result | Parameter values\n");
	    printf("-------------+--------------------------------------------------------\n");
	    break;
	 case 2:
	    printf("  FCN Result | Differences in parameter values from initial\n");
	    printf("-------------+--------------------------------------------------------\n");
                self->myfcn->SetOriginal(self->upar->Params());
	    break;
	 case 3:
	    printf("  FCN Result | Differences in parameter values from the previous\n");
	    printf("-------------+--------------------------------------------------------\n");
                self->myfcn->SetOriginal(self->upar->Params());
	    break;
      }
   }
    self->myfcn->SetPrintMode(self->printMode);

   if (!PyDict_Check(self->values)) {
      PyErr_SetString(PyExc_TypeError, "values must be a dictionary.");
      if (subargs != NULL) Py_DECREF(subargs);
      self->scandepth = 0;
      return NULL;
   }
   
   if (self->scandepth == 0) {
      self->ncalls = 0;

      for (int i = 0;  i < self->npar;  i++) {
	 PyObject *value = PyDict_GetItemString(self->values, self->upar->Name(i));
	 if (value == NULL) {
	    PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from values.", self->upar->Name(i));
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }
	 if (!PyNumber_Check(value)) {
	    PyErr_Format(PyExc_TypeError, "values[\"%s\"] must be a number.", self->upar->Name(i));
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }
	 self->upar->SetValue(i, PyFloat_AsDouble(value));
      }
   }
    self->myfcn->SetOriginal(self->upar->Params());

   int index = -1;
   for (int i = 0;  i < self->npar;  i++) {
      if (strcmp(self->upar->Name(i), PyString_AsString(param)) == 0) {
	 index = i;
      }
   }

   double x = low;
   double stepsize = (high - low) / double(bins);

   if (kwds == 0  ||
       PyDict_GetItemString(kwds, "corners") == NULL  ||
       PyDict_GetItemString(kwds, "corners") == Py_False) {
      x += stepsize / 2.;
   }

   PyObject *output = PyList_New(bins);
   for (int i = 0;  i < bins;  i++) {
      self->upar->SetValue(index, x);
      
      if (subargs == NULL) {
	 double result;
	 try {
	    result = (*self->myfcn)(self->upar->Params());
	    self->ncalls++;
	 }
	 catch (ExceptionDuringMinimization theException) {
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }

	 PyObject *num = PyFloat_FromDouble(result);
	 if (num == NULL) {
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }

	 if (PyList_SetItem(output, i, num) != 0) {
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }

	 // New minimum; update fval and values
	 if (self->fval == Py_None  ||  PyFloat_AsDouble(self->fval) > result) {
	    Py_DECREF(self->fval);
	    self->fval = PyFloat_FromDouble(result);

	    for (int j = 0;  j < self->npar;  j++) {
	       PyObject *value = PyFloat_FromDouble(self->upar->Value(j));
	       if (PyDict_SetItemString(self->values, self->upar->Name(j), value) != 0) {
		  Py_DECREF(value);
		  if (subargs != NULL) Py_DECREF(subargs);
		  self->scandepth = 0;
		  return NULL;
	       }
	       Py_DECREF(value);
	    }
	 }
      }

      else {
	 self->scandepth++;
	 PyObject *subscan = minuit_Minuit_scan(self, subargs, kwds);
	 if (subscan == NULL) {
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }

	 if (PyList_SetItem(output, i, subscan) != 0) {
	    if (subargs != NULL) Py_DECREF(subargs);
	    self->scandepth = 0;
	    return NULL;
	 }
      }

      x += stepsize;
   }

   self->scandepth--;

   if (kwds == 0  ||
       PyDict_GetItemString(kwds, "output") == NULL  ||
       PyDict_GetItemString(kwds, "output") == Py_True) {
      return output;
   }
   else {
      Py_DECREF(output);
      return Py_BuildValue("O", Py_None);
   }
}

static PyObject* minuit_Minuit_matrix(minuit_Minuit* self, PyObject* args, PyObject* kwds) {
   PyObject *correlation = Py_False;
   PyObject *skip_fixed = Py_False;
   static char *kwlist[] = {"correlation", "skip_fixed", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &correlation, &skip_fixed)) {
      PyErr_SetString(PyExc_TypeError, "Two optional arguments: correlation=True (for a correlation, rather than covariance matrix), skip_fixed=True (skips fixed parameters).");
      return NULL;
   }
   if (correlation != Py_True  &&  correlation != Py_False) {
      PyErr_SetString(PyExc_TypeError, "correlation must be True or False.");
      return NULL;
   }
   if (skip_fixed != Py_True  &&  skip_fixed != Py_False) {
      PyErr_SetString(PyExc_TypeError, "skip_fixed must be True or False.");
      return NULL;
   }

   if (!PyDict_Check(self->fixed)) {
      PyErr_SetString(PyExc_TypeError, "fixed must be a dictionary.");
      return NULL;
   }

   if (!PyDict_Check(self->covariance)) {
      PyErr_SetString(PyExc_TypeError, "covariance must be a dictionary.");
      return NULL;
   }

   std::vector<PyObject*> vars;
   for (int i = 0;  i < self->npar;  i++) {
      PyObject *var = PyTuple_GetItem(self->parameters, i);
      PyObject *isfixed = PyDict_GetItem(self->fixed, var);
      if (isfixed == NULL) {
	 PyErr_Format(PyExc_KeyError, "Parameter \"%s\" is missing from the fixed dictionary.", self->upar->Name(i));
	 return NULL;
      }
      if (isfixed != Py_True  &&  isfixed != Py_False) {
	 PyErr_SetString(PyExc_TypeError, "Entries in the fixed dictionary must be True or False.");
	 return NULL;
      }
      if (skip_fixed == Py_False  ||  isfixed == Py_False) {
	 vars.push_back(var);
      }
   }

   PyObject *output = PyTuple_New(vars.size());
   if (output == NULL) {
      return NULL;
   }

   for (unsigned int i = 0;  i < vars.size();  i++) {
      PyObject *line = PyTuple_New(vars.size());
      if (line == NULL) {
	 Py_DECREF(output);
	 return NULL;
      }

      if (PyTuple_SetItem(output, i, line) != 0) {
	 Py_DECREF(output);
	 Py_DECREF(line);
	 return NULL;
      }

      for (unsigned int j = 0;  j < vars.size();  j++) {
	 PyObject *key = Py_BuildValue("OO", vars[i], vars[j]);
	 PyObject *val = PyDict_GetItem(self->covariance, key);
	 Py_DECREF(key);
	 if (val == NULL) {
	    val = PyFloat_FromDouble(0.);
	 }
	 else {
	    Py_INCREF(val);

	    if (correlation == Py_True) {
	       key = Py_BuildValue("OO", vars[i], vars[i]);
	       PyObject *ii = PyDict_GetItem(self->covariance, key);
	       Py_DECREF(key);
	       key = Py_BuildValue("OO", vars[j], vars[j]);
	       PyObject *jj = PyDict_GetItem(self->covariance, key);
	       Py_DECREF(key);
	       
	       if (PyFloat_AsDouble(ii) == 0.  ||  PyFloat_AsDouble(jj) == 0.) {
		  PyErr_SetString(PyExc_MinuitError, "A diagonal element in the correlation matrix is zero.");
		  Py_DECREF(output);
		  Py_DECREF(val);
		  return NULL;
	       }

	       double v = PyFloat_AsDouble(val);
	       Py_DECREF(val);
	       val = PyFloat_FromDouble(v / sqrt(PyFloat_AsDouble(ii)) / sqrt(PyFloat_AsDouble(jj)));
	    }
	 }
	 
	 if (PyTuple_SetItem(line, j, val) != 0) {
	    Py_DECREF(output);
	    return NULL;
	 }
      }
   }
   
   return output;
}

double MyFCN::operator()(const std::vector<double>& par) const {
    int argsize = m_npar;
    if (m_self){
        ++argsize;
    }
    PyObject *args = PyTuple_New(argsize);
   if (args == NULL) {
      throw ExceptionDuringMinimization();
   }

   int i = 0;
   std::vector<double>::const_iterator pend = par.end();
    if (m_self){
        PyTuple_SetItem(args, i, m_self);
        ++i;
    }
   for (std::vector<double>::const_iterator p = par.begin();  p != pend;  ++p) {
      PyObject *arg = PyFloat_FromDouble(*p);
      if (arg == NULL) {
	 Py_DECREF(args);
	 throw ExceptionDuringMinimization();
      }
      if (PyTuple_SetItem(args, i, arg) != 0) {
	 Py_DECREF(args);
	 Py_DECREF(arg);
	 throw ExceptionDuringMinimization();
      }
      i++;
   }

    Py_XINCREF(m_self);
   PyObject *result = PyObject_CallObject(m_fcn, args);
   Py_DECREF(args);
   if (result == NULL) {
      throw ExceptionDuringMinimization();
   }
   if (!PyNumber_Check(result)) {
      PyErr_SetString(PyExc_TypeError, "The function must return a number.");
      Py_DECREF(result);
      throw ExceptionDuringMinimization();
   }

   double res = PyFloat_AsDouble(result);
   Py_DECREF(result);

   if (m_printMode > 0) {
      switch (m_printMode) {
	 case 1:
	    printf("%12g |", res);
	    for (int i = 0;  i < m_npar;  i++) printf(" %12g", par[i]);
	    printf("\n");
	    break;

	 case 2:
	    printf("%12g |", res);
	    for (int i = 0;  i < m_npar;  i++) printf(" %12g", par[i]-m_original[i]);
	    printf("\n");
	    break;
	    
	 case 3:
	    static std::vector<double> last;
	    if (last.size() == 0) {
	       last = m_original;
	    }
	    printf("%12g |", res);
	    for (int i = 0;  i < m_npar;  i++) printf(" %12g", par[i]-last[i]);
	    printf("\n");
	    last = par;
	    break;
      }
   }

   return res;
}

static PyObject* minuit_machine_precision() {
   MnMachinePrecision mp;
   return Py_BuildValue("d", mp.Eps());
}

static PyMethodDef minuit_methods[] = {
   {"machine_precision", (PyCFunction)(minuit_machine_precision), METH_NOARGS, "Return the current machine precision.  In the old MINUIT, this function was called EPS()."},
   {NULL}
};

// Python 3 module initialization
static PyObject *
moduleinit(void) {
   PyObject *m;

   minuit_MinuitType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&minuit_MinuitType) < 0) return NULL;

    MOD_DEF(m, "minuit2", "Interface to ROOT MINUIT2", minuit_methods);
    if (m == NULL){
        return NULL;
    }
   Py_INCREF(&minuit_MinuitType);
    PyModule_AddObject(m, "Minuit2", (PyObject*)(&minuit_MinuitType));

    PyExc_MinuitError = PyErr_NewException("minuit2.MinuitError", NULL, NULL);
    if (PyExc_MinuitError == NULL) return NULL;
   Py_INCREF(PyExc_MinuitError);
   PyModule_AddObject(m, "MinuitError", PyExc_MinuitError);
    return m;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initminuit2(void)
{
    moduleinit();
}
#else
PyMODINIT_FUNC PyInit_minuit2(void)
{
    return moduleinit();
}
#endif
