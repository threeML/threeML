import astropy.units as u
import pytest
from astromodels import Model, PointSource, Powerlaw

from threeML import plot_spectra
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.io.logging import setup_logger
from threeML.io.network import internet_connection_is_active
from threeML.utils.data_builders.fermi.lat_transient_builder import (
    TransientLATDataBuilder,
)
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset

log = setup_logger(__name__)

skip_if_internet_is_not_available = pytest.mark.skipif(
    not internet_connection_is_active(), reason="No active internet connection"
)


@skip_if_internet_is_not_available
def test_make_LAT_dataset(tmp_path):

    _ = pytest.importorskip("GtApp")
    _ = pytest.importorskip("GtBurst")
    trigger_time = 243216766
    ra = 119.84717
    dec = -56.638333
    radius = 10.0
    zmax = 110.0
    thetamax = 180.0
    irf = "p8_transient020e"
    datarepository = tmp_path / "FermiData"

    myLATdataset = LAT_dataset()

    myLATdataset.make_LAT_dataset(
        ra,
        dec,
        radius=radius + 10,
        trigger_time=trigger_time,
        tstart=-10,
        tstop=100,
        data_type="Extended",
        destination_directory=datarepository,
        Emin=30.0,
        Emax=1000000.0,
    )

    myLATdataset.extract_events(radius, zmax, irf, thetamax, strategy="time")

    analysis_builder = TransientLATDataBuilder(
        myLATdataset.grb_name,
        outfile=myLATdataset.grb_name,
        roi=radius,
        tstarts="0,10",
        tstops="10,100",
        irf=irf,
        galactic_model="template",
        particle_model="isotr template",
        datarepository=datarepository,
    )
    analysis_builder.display()

    analysis_builder.run(include_previous_intervals=True)

    LAT_Like_plugins = analysis_builder.to_LATLike()

    spectrum = Powerlaw()
    spectrum.piv = 1e5  # 100 MeV
    results = []
    for myplug in LAT_Like_plugins:
        data = DataList(myplug)

        test_source = PointSource("test_source", ra, dec, spectrum)

        my_model = Model(test_source)

        jl = JointLikelihood(my_model, data)

        jl.fit()

        flux = jl.results.get_flux(100.0 * u.MeV, 10.0 * u.GeV)
        print(flux)
        results.append(jl.results)
        # energies = np.logspace(1, 4, 100) * u.MeV
        # differential_flux = my_model.test_source(energies)
        # plt.loglog(energies, differential_flux)

    plot_spectra(
        *results,
        flux_unit="erg2/(cm2 s keV)",
        energy_unit="MeV",
        ene_min=10,
        ene_max=10e4,
    )
