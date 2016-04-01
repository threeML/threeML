#include "FixedPointSource.h"
#include "FakePlugin.h"
#include <iostream>
#include <math.h>

int main(int argc, char **argv) {
  //Since I don't specify the parameters for the FixedPointSource
  //the values for the Crab will be used (see FixedPointSource.h)
  threeML::FixedPointSource src;
  
  //Print info on the source
  src.describe();
  
  //A fake plugin which just showcase how to get and use the
  //ModelInterface instance
  //(it will also print some info during the constructor)
  threeML::FakePlugin fp(&src);
  
  //Create a certain number of energy bins
  //to be used in the flux calculation
  int n = 20;
  fp.createEnergies(n);
  
  //This will compute and print the flux
  //from the FixedPointSource at the energies
  //defined above
  fp.go();
  
}
