from threeML.bayesian.sampler import UnitCubeSampler
from threeML import threeML_config


class UltraNestSampler(UnitCubeSampler):

    def __init__(self, likelihood_model, data_list, **kwargs):

        super(UltraNestSampler, self).__init__(likelihood_model, data_list, **kwargs)


    def setup(self):
        pass
