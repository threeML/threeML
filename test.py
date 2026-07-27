# Source - https://stackoverflow.com/a/60679407
# Posted by BarryPye, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-22, License - CC BY-SA 4.0

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--log",
    default="warning",
    help=("Provide logging level. " "Example --log debug', default='warning'"),
)

options = parser.parse_args()
levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
level = levels.get(options.log.lower())
if level is None:
    raise ValueError(
        f"log level given: {options.log}"
        f" -- must be one of: {' | '.join(levels.keys())}"
    )
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)

from threeML import plot_spectra
from threeML.analysis_results import load_analysis_results
import matplotlib.pyplot as plt

ar = load_analysis_results("hess_fermi_pynch.h5")
try:
    ar.corner_plot_cc()
    plt.show()
except:
    pass

fig = plot_spectra(
    ar,
    flux_unit="erg/(cm2 s)",
    ene_min=0.1,
    ene_max=1e10,
    energy_unit="keV",
    sources_to_use=["g21509"],
    use_components=True,
    components_to_use=["ic", "total"],
)
ax = fig.gca()
ax.set_ylim(1e-20, 1e-7)
plt.show()
