import matplotlib.pyplot as plt

from threeML.io.step_plot import step_plot
from threeML.config.config import threeML_config


# Fermi specific plots

def fermi_light_curve_plot(time_bins, cnts, bkg, width, selection, bkg_selections, instrument):
    fig, ax = plt.subplots()

    max_cnts = max(cnts / width)
    top = max_cnts + max_cnts * .2
    min_cnts = min(cnts[cnts > 0] / width[cnts > 0])
    bottom = min_cnts - min_cnts * .05
    mean_time = map(np.mean, time_bins)

    all_masks = []

    # purple: #8da0cb

    step_plot(time_bins, cnts / width, ax,
              color=threeML_config[instrument]['lightcurve color'], label="Light Curve")

    for tmin, tmax in selection:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:
            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config[instrument]['selection color'],
                      fill=True,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config[instrument]['selection color'],
              fill=True,
              fill_min=min_cnts, label="Selection")

    all_masks = []
    for tmin, tmax in bkg_selections:
        tmp_mask = np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)

        all_masks.append(tmp_mask)

    if len(all_masks) > 1:

        for mask in all_masks[1:]:

            step_plot(time_bins[mask], cnts[mask] / width[mask], ax,
                      color=threeML_config[instrument]['background selection color'],
                      fill=True,
                      fillAlpha=.4,
                      fill_min=min_cnts)

    step_plot(time_bins[all_masks[0]], cnts[all_masks[0]] / width[all_masks[0]], ax,
              color=threeML_config[instrument]['background selection color'],
              fill=True,
              fill_min=min_cnts, fillAlpha=.4, label="Bkg. Selections")

    ax.plot(mean_time, bkg, threeML_config[instrument]['background color'], lw=2., label="Background")

    # ax.fill_between(selection, bottom, top, color="#fc8d62", alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom, top)
    ax.set_xlim(time_bins.min(), time_bins.max())
    ax.legend()
