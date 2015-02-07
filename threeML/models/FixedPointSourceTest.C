#include <threeML/FixedPointSource.h>
#include <iostream>
#include <math.h>

int main(int argc, char **argv) {
  threeML::FixedPointSource src;
  
  src.describe();
    
  std::cout << std::endl << "Spectrum from 1 GeV to 1 TeV:" 
            << std::endl << std::endl;
  
  double logmin = 6.0;
  double logmax = 9.0;
  double emin = pow(10.0,logmin); //MeV
  double emax = pow(10.0,logmax); //MeV
  int nbins = 20;
  
  std::vector<double> energies(nbins);
  int i;
  double step = (logmax-logmin)/nbins;
  
  for(i=0; i < nbins; ++i) 
  {
    energies[i] = pow(10.0,(logmin + i*step));
  }
  
  std::vector<double> fluxes = src.getPointSourceFluxes(0,energies);
  
  for(i=0; i < nbins; ++i) 
  {
    std::cout << energies[i] << " MeV -> " << fluxes[i] << std::endl;
  }
  
  std::cout << std::endl << "Number of point sources in model: " 
            << src.getNumberOfPointSources() << std::endl;
  
  std::cout << std::endl << "Number of ext. sources in model: "
            << src.getNumberOfExtendedSources() << std::endl;
  
  double ra,dec;
  src.getPointSourcePosition(0,&ra,&dec);
  
  std::cout << "getPointSourcePosition() -> (R.A., Dec) = (" 
            << ra << ", " << dec << ")" << std::endl;
  
  
}

