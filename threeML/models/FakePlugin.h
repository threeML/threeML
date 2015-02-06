#include <vector>

namespace threeML {

class ModelInterface;


  class FakePlugin {
    public:
      FakePlugin(const ModelInterface &model) : mi(model) 
        {
          std::cout << std::endl << "Model received:" << std::endl;
          
          int n = mi.getNumberOfPointSources();
          std::cout << "Number of point sources: " << n << std::endl;    
          
          int i;
          for(i=0; i < n; ++i) 
            {
              std::cout << "Point source " << i << " (" << mi.getPointSourceName(i) << ")" << std::endl;
              
              double ra;
              double dec;
              mi.getPointSourcePosition(i,&ra,&dec);
              
              std::cout << "  R.A. = " << ra << ", Dec. = " << dec << std::endl; 
            }
          
           std::cout << std::endl;
        }
      
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
          
          int i;
          int n = mi.getNumberOfPointSources();
                    
          for(i=0; i < n; ++i) 
            {
              std::cout << "Flux for " << mi.getPointSourceName(i) << ":" << std::endl;
              
              std::vector<double> fluxes = mi.getPointSourceFluxes(i,m_energies);
              
              int j;
              for(j=0; j < m_energies.size(); ++j) 
                {
                  std::cout << m_energies[j] << " MeV -> " << fluxes[j] << std::endl;
                }
              std::cout << std::endl;
            }                   
      }
      
    private:
      ModelInterface mi;
      std::vector<double> m_energies;
  
  };

}
