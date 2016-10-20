from threeML import *



data_dir = os.path.join('../examples/gbm','bn080916009')

src_selection = "0.-10."


# We start out with a bad background interval to demonstrate a few features

nai3 = FermiGBMTTELike('NAI3',
                         os.path.join(data_dir, "glg_tte_n3_bn080916009_v01.fit.gz"),
                         "-10-0, 100-200",
                         src_selection,
                         rsp_file=os.path.join(data_dir, "glg_cspec_n3_bn080916009_v07.rsp"),poly_order=0)

nai3.set_active_measurements("10-900")


nai3.write_pha('test',overwrite=True)
