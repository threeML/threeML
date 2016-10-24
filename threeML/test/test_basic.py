from threeML import *
from threeML.plugins.OGIPLike import OGIPLike

def test_a_basic_analysis_from_start_to_finish():

    # In[2]:

    triggerName = 'bn090217206'
    ra = 204.9
    dec = -8.4

    #Data are in the current directory

    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples'))

    #Create an instance of the GBM plugin for each detector
    #Data files
    obsSpectrum = os.path.join( datadir, "bn090217206_n6_srcspectra.pha{1}" )
    bakSpectrum = os.path.join( datadir, "bn090217206_n6_bkgspectra.bak{1}" )
    rspFile     = os.path.join( datadir, "bn090217206_n6_weightedrsp.rsp{1}" )

    #Plugin instance
    NaI6 = OGIPLike( "NaI6", obsSpectrum, bakSpectrum, rspFile )

    #Choose energies to use (in this case, I exclude the energy
    #range from 30 to 40 keV to avoid the k-edge, as well as anything above
    #950 keV, where the calibration is uncertain)
    NaI6.set_active_measurements( "10.0-30.0", "40.0-950.0" )


    # In[3]:

    #This declares which data we want to use. In our case, all that we have already created.

    data_list = DataList( NaI6 )


    # In[4]:

    powerlaw = Powerlaw()


    # In[5]:

    GRB = PointSource( triggerName, ra, dec, spectral_shape=powerlaw )


    # In[6]:

    model = Model( GRB )


    # In[7]:

    jl = JointLikelihood( model, data_list, verbose=False )

    fit_results, like_frame = jl.fit()

    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.K'] - 2.531028) < 1e-2
    assert abs(fit_results['value']['bn090217206.spectrum.main.Powerlaw.index'] + 1.1831566000728451) < 1e-2

    # In[8]:

    #bn090217206.spectrum.main.Powerlaw.K	2.57 -0.19 +0.22	1 / (cm2 keV s)
    #bn090217206.spectrum.main.Powerlaw.index	-1.185 -0.015 +0.014


    res = jl.get_errors()


    # In[9]:

    res = jl.get_contours(powerlaw.index,-1.3,-1.1,20)


    # In[10]:

    res = jl.get_contours(powerlaw.index,-1.25,-1.1,60,powerlaw.K,1.8,3.4,60)


    # In[11]:

    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0, upper_bound=10)

    bayes = BayesianAnalysis(model, data_list)


    # In[12]:

    samples = bayes.sample(n_walkers=50,burn_in=10, n_samples=10)


    # In[13]:

    fig = bayes.corner_plot()


# In[ ]:



