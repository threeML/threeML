#include <vector>

namespace threeML {

class ModelInterface;


  class FakePlugin {
    public:
      FakePlugin(const ModelInterface &model) : mi(model) {}
      
      void createEnergies(int n) 
          {
            m_energies.resize(n,0);
            int i;
            for(i=0; i < m_energies.size(); ++i) 
            {
              m_energies[i] = (i+1)*100.0;
            }
          }
      
      void go() {
          
          std::vector<double> fluxes = mi.getPointSourceFluxes(0,m_energies);
          
          /*
          int i;
          unsigned long size = energies.size();
          
          for (i=0; i < size; ++i)
          {
            std::cout << "Flux at " << energies[i] << " MeV is " << fluxes[i] << std::endl;
          }*/
                   
      }
      
    private:
      ModelInterface mi;
      std::vector<double> m_energies;
  
  };

}
