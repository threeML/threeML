from threeML.bayesian.sampler import MCMCSampler
from threeML import threeML_config


class EmceeSampler(MCMCSampler):

    def __init__(self, likelihood_model, data_list, **kwargs):

        super(EmceeSampler, self).__init__(likelihood_model, data_list, **kwargs)


    def setup(self):
        pass
