// minuit2.h is Copyright (C) 2008 Johann Cohen-Tanugi <cohen@slac.stanford.edu>
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

#include <Python.h>
#include <structmember.h>
#include <vector>
#include <sstream>
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnSimplex.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnCross.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/ContoursError.h"

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

using namespace ROOT::Minuit2;

class ExceptionDuringMinimization {
   public:
      ExceptionDuringMinimization() {}
      ~ExceptionDuringMinimization() {}
};

class MyFCN: public FCNBase {
   public:
      MyFCN(PyObject *fcn, PyObject *self, int npar): m_fcn(fcn), m_self(self), m_npar(npar) { };
      double operator()(const std::vector<double>& par) const;

      double Up() const { return m_up; }
      void SetUp(double up) { m_up = up; }
      void SetPrintMode(int printMode) { m_printMode = printMode; }
      void SetOriginal(std::vector<double> par) { m_original = par; }
      
   private:
      PyObject *m_fcn;
      PyObject *m_self;
      int m_npar;
      double m_up;
      int m_printMode;
      std::vector<double> m_original;
};

typedef struct {
      PyObject_HEAD

      MyFCN *myfcn;
      MnUserParameters *upar;
      FunctionMinimum *min;
      int scandepth;

      int npar;
      PyObject *maxcalls;
      double tol;
      int strategy;
      double up;
      int printMode;
      PyObject *fixed;
      PyObject *limits;
      PyObject *values;
      PyObject *args;
      PyObject *errors;
      PyObject *merrors;
      PyObject *covariance;

      PyObject *fcn;
      PyObject *self;
      PyObject *fval;
      int ncalls;
      PyObject *edm;
      PyObject *parameters;
} minuit_Minuit;

static int minuit_Minuit_init(minuit_Minuit* self, PyObject* args, PyObject* kwds);
static int minuit_Minuit_dealloc(minuit_Minuit* self);
bool minuit_prepare(minuit_Minuit *self, int &maxcalls, std::vector<std::string> &floating);
static PyObject* minuit_Minuit_simplex(minuit_Minuit* self);
static PyObject* minuit_Minuit_migrad(minuit_Minuit* self);
static PyObject* minuit_Minuit_hesse(minuit_Minuit* self);
static PyObject* minuit_Minuit_minos(minuit_Minuit* self, PyObject* args);
static PyObject* minuit_Minuit_contour(minuit_Minuit* self, PyObject* args, PyObject* kwds);
static PyObject* minuit_Minuit_scan(minuit_Minuit* self, PyObject* args, PyObject* kwds);
static PyObject* minuit_Minuit_matrix(minuit_Minuit* self, PyObject* args, PyObject* kwds);
static PyObject* minuit_machine_precision();
