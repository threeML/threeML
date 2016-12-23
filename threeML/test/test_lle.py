import pytest
import numpy as np
import os

__author__ = 'grburgess'

from threeML.plugins.FermiLATLLELike import FermiLATLLELike, BinningMethodError
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


def is_within_tolerance(truth, value, relative_tolerance=0.01):
    assert truth != 0

    if abs((truth - value) / truth) <= relative_tolerance:

        return True

    else:

        return False


def examine_bins(bins, real_start, real_stop, expected_number_of_bins):
    assert len(bins) == 2

    starts, stops = bins



    # check that the start and stop make sense

    assert np.round(starts[0], decimals=0) >= real_start
    assert np.round(stops[-1], decimals=0) <= real_stop

    # test bin ordering

    for x, y in zip(starts, stops):

        assert x < y

    # test bin length

    assert len(starts) == expected_number_of_bins
    assert len(stops) == expected_number_of_bins


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


def test_gbm_lle_constructor():
    with within_directory(__example_dir):
        data_dir = 'lat'

        src_selection = "0.-10."

        nai3 = FermiLATLLELike('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                               os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                               "-100-0, 100-200",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"), poly_order=-1)

        assert nai3.name == 'lle'

        assert nai3._active_interval == ('0.-10.',)
        assert nai3._startup == False
        assert nai3._verbose == True
        assert nai3.background_noise_model == 'gaussian'

        nai3.view_lightcurve()

        nai3.view_lightcurve(energy_selection="500000-100000")

        nai3.background_poly_order = 2

        nai3.background_poly_order = -1

        nai3.set_active_measurements("50000-1000000")

        nai3.set_active_time_interval("0-10")

        nai3.set_background_interval("-150-0", "100-250")

        #nai3.set_background_interval("-15-0", "100-150", unbinned=False)


def test_lle_binning():
    with within_directory(__example_dir):
        data_dir = 'lat'

        src_selection = "0.-10."

        nai3 = FermiLATLLELike('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                               os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                               "-100-0, 100-200",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"), poly_order=-1)
        # should not have bins yet



        with pytest.raises(AttributeError):
            nai3.bins

        with pytest.raises(AttributeError):
            nai3.text_bins

        # First catch the errors


        # This is without specifying the correct options name





        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='constant')

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='significance')

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='constant', p0=.1)

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='significance', dt=1)

        # now incorrect options

        with pytest.raises(RuntimeError):
            nai3.create_time_bins(start=0, stop=10, method='not_a_method')

        # Now test values



        nai3.create_time_bins(start=0, stop=10, method='constant', dt=1)

        assert len(nai3.text_bins) == 10

        examine_bins(nai3.bins, 0, 10, 10)

        nai3.create_time_bins(start=0, stop=10, method='bayesblocks', p0=.1)

        examine_bins(nai3.bins, 0, 10, 6)

        nai3.create_time_bins(start=0, stop=10, method='significance', sigma=10)

        examine_bins(nai3.bins, 0, 10, 31)

        nai3.view_lightcurve(use_binner=True)

        nai3.write_pha_from_binner("test_binner", overwrite=True)

        ogips = nai3.get_ogip_from_binner()


def test_gbm_lle_joint_likelihood_fitting():
    with within_directory(__example_dir):
        data_dir = 'lat'

        src_selection = "0.-70."

        nai3 = FermiLATLLELike('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                               os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                               "-100-0, 100-200",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"), poly_order=-1)

        nai3.set_active_measurements("50000-100000")

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

            # res, _ = jl.fit()
            #
            # assert jl.current_minimum is not None
            #
            # assert jl.minimizer is not None
            # assert jl.ncalls != 1


def test_gbm_lle_bayesian_fitting():
    with within_directory(__example_dir):
        data_dir = 'lat'

        src_selection = "0.-10."

        nai3 = FermiLATLLELike('lle', os.path.join(data_dir, "gll_lle_bn080916009_v10.fit"),
                               os.path.join(data_dir, "gll_pt_bn080916009_v10.fit"),
                               "-100-0, 100-200",
                               src_selection,
                               rsp_file=os.path.join(data_dir, "gll_cspec_bn080916009_v10.rsp"), poly_order=-1)



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


            # assert jl.ncalls != 1
