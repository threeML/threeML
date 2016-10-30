import pytest
import numpy as np
import os

__author__ = 'drjfunk'

from threeML.plugins.FermiGBMTTELike import FermiGBMTTELike
from threeML.data_list import DataList
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from astromodels.model import Model
from astromodels.functions.functions import Powerlaw, Exponential_cutoff
from astromodels.sources.point_source import PointSource
from astromodels.sources.extended_source import ExtendedSource
from astromodels.functions.functions import Log_uniform_prior, Uniform_prior

from threeML.io.file_utils import within_directory

__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))

__example_dir = os.path.join(__this_dir__, '../../examples')


class AnalysisBuilder(object):
    def __init__(self, plugin):

        self._plugin = plugin

        self._shapes = {}
        self._shapes['normal'] = Powerlaw()
        self._shapes['add'] = Powerlaw() + Powerlaw()
        self._shapes['mult'] = Powerlaw() * Exponential_cutoff()
        self._shapes['crazy'] = Exponential_cutoff() * (Powerlaw() + Powerlaw())

    @property
    def keys(self):

        return self._shapes.keys()

    def set_priors(self):

        key = 'normal'

        self._shapes[key].K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index.set_uninformative_prior(Uniform_prior)

        key = 'add'

        self._shapes[key].K_1.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index_1.set_uninformative_prior(Uniform_prior)
        self._shapes[key].K_2.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index_2.set_uninformative_prior(Uniform_prior)

        key = 'mult'

        self._shapes[key].K_1.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index_1.set_uninformative_prior(Uniform_prior)
        self._shapes[key].K_2.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].xc_2.prior = Log_uniform_prior(lower_bound=1e0, upper_bound=1e2)

        key = 'crazy'

        self._shapes[key].K_1.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].xc_1.prior = Log_uniform_prior(lower_bound=1e0, upper_bound=1e2)
        self._shapes[key].K_2.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index_2.set_uninformative_prior(Uniform_prior)
        self._shapes[key].K_3.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e2)
        self._shapes[key].index_3.set_uninformative_prior(Uniform_prior)

    def build_point_source_jl(self):

        data_list = DataList(self._plugin)

        jls = {}

        for key in self._shapes.keys():
            ps = PointSource('test', 0, 0, spectral_shape=self._shapes[key])
            model = Model(ps)
            jls[key] = JointLikelihood(model, data_list)

        return jls

    def build_point_source_bayes(self):

        data_list = DataList(self._plugin)

        bayes = {}

        for key in self._shapes.keys():
            ps = PointSource('test', 0, 0, spectral_shape=self._shapes[key])
            model = Model(ps)

            bayes[key] = BayesianAnalysis(model, data_list)

        return bayes


def test_gbm_tte_constructor():
    with within_directory(__example_dir):
        data_dir = os.path.join('gbm', 'bn080916009')

        src_selection = "0.-10."

        nai3 = FermiGBMTTELike('NAI3',
                               os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                               "-10-0, 100-150",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"), poly_order=-1)

        assert nai3.name == 'NAI3'

        assert nai3._active_interval == ('0.-10.',)
        assert nai3._startup == False
        assert nai3._verbose == True
        assert nai3.background_noise_model == 'gaussian'


def test_gbm_tte_joint_likelihood_fitting():
    with within_directory(__example_dir):
        data_dir = os.path.join('gbm', 'bn080916009')

        src_selection = "0.-10."

        nai3 = FermiGBMTTELike('NAI3',
                               os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                               "-10-0, 100-150",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"), poly_order=-1)

        ab = AnalysisBuilder(nai3)

        jls = ab.build_point_source_jl()

        for key in ['normal']:

            jl = jls[key]

            assert jl.analysis_type == 'mle'

            # no COV yet
            with pytest.raises(RuntimeError):

                _ = jl.covariance_matrix

            with pytest.raises(RuntimeError):

                _ = jl.correlation_matrix

            assert jl.current_minimum is None
            assert jl.minimizer_in_use == ('MINUIT', None, None)
            assert jl.minimizer is None
            assert jl.ncalls == 0
            assert jl.verbose == False

            res, _ = jl.fit()

            assert jl.current_minimum is not None

            assert jl.minimizer is not None
            assert jl.ncalls != 1


def test_gbm_tte_bayesian_fitting():
    with within_directory(__example_dir):
        data_dir = os.path.join('gbm', 'bn080916009')

        src_selection = "0.-10."

        nai3 = FermiGBMTTELike('NAI3',
                               os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                               "-10-0, 100-150",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"), poly_order=-1)

        ab = AnalysisBuilder(nai3)
        ab.set_priors()
        bayes = ab.build_point_source_bayes()

        for key in ['normal']:

            bb = bayes[key]

            assert bb.analysis_type == 'bayesian'
            assert bb._samples is None
            assert bb._raw_samples is None
            assert bb._sampler is None
            assert bb._log_like_values is None

            n_walk = 10
            n_samp = 100

            n_samples = n_walk * n_samp

            samples = bb.sample(n_walkers=n_walk, burn_in=10, n_samples=n_samp)

            for key_2 in samples.keys():
                assert samples[key_2].shape[0] == n_samples

            assert len(bb.log_like_values) == n_samples
            # assert len(bb.log_probability_values) == n_samples

            assert bb.raw_samples.shape == (n_samples, 2)




            # assert bb.current_minimum is None
            # assert jl.minimizer_in_use == ('MINUIT', None)
            # assert jl.minimizer is None
            # assert jl.ncalls == 0
            # assert jl.verbose == False

            # res, _ = jl.fit()

            # assert jl.current_minimum is not None

            # assert jl.minimizer is not None
            # assert jl.ncalls != 1
