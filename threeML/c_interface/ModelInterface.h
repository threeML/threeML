//Author: G.Vianello (giacomov@slac.stanford.edu)

//Interface class for the likelihood model
//Derived class must implement all these methods

// *************  DO NOT CHANGE ANY DETAIL HERE FOR ANY REASON *********************

// Changing something here will get this header out of sync with the copy contained
// in 3ML and give you weird crashes when using LiFF from 3ML.

// *********************************************************************************

#define INTERFACE_VERSION 1.0

#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H

#include <vector>
#include <string>

namespace threeML {

class ModelInterface {

 public:

  virtual int getInterfaceVersion() const { return INTERFACE_VERSION; }

  virtual ~ModelInterface() {};

  //The use of "const" at the end of the method declaration promises
  //to the compiler that the method will not change the class
  //(in other words, these are all read-only methods)

  //Point source interface

  virtual int getNumberOfPointSources() const =0;

  virtual void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const =0;

  //Fluxes are differential fluxes in MeV^-1 cm^-1 s^-1
  virtual std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies) const =0;

  virtual std::string getPointSourceName(int srcid) const =0;

  //Extended source interface

  virtual int getNumberOfExtendedSources() const =0;

  virtual std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec,
                                                      std::vector<double> energies) const =0;

  virtual std::string getExtendedSourceName(int srcid) const =0;

  virtual bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const =0;

  virtual void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                           double *j2000_ra_max,
                                           double *j2000_dec_min,
                                           double *j2000_dec_max) const =0;

};

// This EmptyModelInterface implements an empty model (zero point sources and zero extended sources).
// It is useful for testing.

class EmptyModelInterface: public ModelInterface {

 public:

  EmptyModelInterface() { }

  int getNumberOfPointSources() const { return 0; }

  void getPointSourcePosition(int srcid, double *j2000_ra, double *j2000_dec) const { }

  //Fluxes are differential fluxes in MeV^-1 cm^-1 s^-1
  std::vector<double> getPointSourceFluxes(int srcid, std::vector<double> energies) const {
    return energies;
  }

  std::string getPointSourceName(int srcid) const { return "none"; }

  //Extended source interface

  int getNumberOfExtendedSources() const { return 0; }

  std::vector<double> getExtendedSourceFluxes(int srcid, double j2000_ra, double j2000_dec,
                                              std::vector<double> energies) const {
    return energies;
  }

  std::string getExtendedSourceName(int srcid) const { return "none"; }

  bool isInsideAnyExtendedSource(double j2000_ra, double j2000_dec) const { return false; }

  void getExtendedSourceBoundaries(int srcid, double *j2000_ra_min,
                                   double *j2000_ra_max,
                                   double *j2000_dec_min,
                                   double *j2000_dec_max) const { }

};

}

#endif
