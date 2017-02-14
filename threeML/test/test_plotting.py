import pytest
from threeML import *
from threeML.plugins.OGIPLike import OGIPLike
from threeML.utils.fitted_objects.fitted_point_sources import InvalidUnitError
from threeML.utils.binner import NotEnoughData



def test_OGIP_plotting():
    # In[2]:

    triggerName = 'bn090217206'
    ra = 204.9
    dec = -8.4

    # Data are in the current directory

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples'))

    # Create an instance of the GBM plugin for each detector
    # Data files
    obsSpectrum = os.path.join(datadir, "bn090217206_n6_srcspectra.pha{1}")
    bakSpectrum = os.path.join(datadir, "bn090217206_n6_bkgspectra.bak{1}")
    rspFile = os.path.join(datadir, "bn090217206_n6_weightedrsp.rsp{1}")

    # Plugin instance
    NaI6 = OGIPLike("NaI6", obsSpectrum, bakSpectrum, rspFile)

    # Choose energies to use (in this case, I exclude the energy
    # range from 30 to 40 keV to avoid the k-edge, as well as anything above
    # 950 keV, where the calibration is uncertain)
    NaI6.set_active_measurements("10.0-30.0", "40.0-950.0")

    # OGIP channel plotting

    NaI6.view_count_spectrum(plot_errors=True, show_bad_channels=True)

    NaI6.view_count_spectrum(plot_errors=False, show_bad_channels=True)

    NaI6.view_count_spectrum(plot_errors=True, show_bad_channels=False)

    NaI6.view_count_spectrum(plot_errors=False, show_bad_channels=False)

    # In[3]:

    # This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList(NaI6)

    # In[4]:

    powerlaw = Powerlaw()

    # In[5]:

    GRB = PointSource(triggerName, ra, dec, spectral_shape=powerlaw)

    # In[6]:

    model = Model(GRB)

    # In[7]:

    jl = JointLikelihood(model, data_list, verbose=False)

    fit_results, like_frame = jl.fit()

    _ = display_spectrum_model_counts(jl)

    _ = display_spectrum_model_counts(jl, data=('NaI6'))

    _ = display_spectrum_model_counts(jl, data=('wrong'))

    _ = display_spectrum_model_counts(jl, min_rate=1E-8)

    with pytest.raises(NotEnoughData):
        _ = display_spectrum_model_counts(jl, min_rate=1E8)
