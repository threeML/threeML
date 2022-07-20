# import threeML
# import pdb;pdb.set_trace()
from lat_transient_builder import TransientLATDataBuilder

lt = TransientLATDataBuilder(
    triggername="bn080916009",
    outfile="analysis",
    roi=5.0,
    tstarts="0.",
    tstops="1528.75",
    irf="p8_transient010e",
    galactic_model="template (fixed norm.)",
    particle_model="isotr template",
    zmax=105.0,
    emin=65.0,
    emax=100000.0,
    ra=119.88999939,
    dec=-56.7000007629,
    liketype="binned",
    log_bins="1., 10000., 30",
)  # bin_file = ''

lt.run()
# import pdb;pdb.set_trace()
