import pytest
from threeML import *
from threeML.plugins.OGIPLike import OGIPLike
from threeML.utils.initalize_testing import initialize_testing

initialize_testing()

def get_data(id):

    # Data are in the current directory

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples'))

    # Create an instance of the GBM plugin for each detector
    # Data files
    obs_spectrum = os.path.join(datadir, "bn090217206_n6_srcspectra.pha{1}")
    bakSpectrum = os.path.join(datadir, "bn090217206_n6_bkgspectra.bak{1}")
    rspFile = os.path.join(datadir, "bn090217206_n6_weightedrsp.rsp{1}")

    # Plugin instance
    NaI6 = OGIPLike("NaI6", obs_spectrum, bakSpectrum, rspFile)

    # Choose energies to use (in this case, I exclude the energy
    # range from 30 to 40 keV to avoid the k-edge, as well as anything above
    # 950 keV, where the calibration is uncertain)
    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    NaI6.display_rsp()

    # This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList(NaI6)

    return data_list


def get_model(id):

    triggerName = 'bn090217206'
    ra = 204.9
    dec = -8.4

    spectrum = Powerlaw() * Line()

    GRB = PointSource(triggerName, ra, dec, spectral_shape=spectrum)

    spectrum.a_2 = 1.0
    spectrum.b_2 = 0.0

    model = Model(GRB)

    return model


def test_joint_likelihood_set():

    jlset = JointLikelihoodSet(data_getter=get_data, model_getter=get_model, n_iterations=10)

    jlset.go(compute_covariance=False)


def test_joint_likelihood_set_parallel():

    jlset = JointLikelihoodSet(data_getter=get_data, model_getter=get_model, n_iterations=10)

    with parallel_computation():

        res = jlset.go(compute_covariance=False)

    print res


