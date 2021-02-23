from threeML import *


__this_dir__ = os.path.join(os.path.abspath(os.path.dirname(__file__)))


data_dir = os.path.join(__this_dir__, "../../examples/gbm", "bn080916009")

src_selection = "0.-10."


# We start out with a bad background interval to demonstrate a few features

nai3 = FermiGBMTTELike(
    "NAI3",
    os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
    "-10-0, 100-200",
    src_selection,
    rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"),
    poly_order=0,
)

nai3.set_active_measurements("10-900")

pl = Powerlaw()

ps = PointSource("test", 0, 0, spectral_shape=pl)

model = Model(ps)

dl = DataList(nai3)

jl = JointLikelihood(data_list=dl, likelihood_model=model)

pl.K = 1e2
pl.index = -2

sim_data = nai3.get_simulated_dataset("sim")

sim_data.write_pha("test", overwrite=True)
