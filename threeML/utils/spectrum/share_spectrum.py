import numpy as np

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.io.logging import setup_logger
log = setup_logger(__name__)


class ShareSpectrum(object):
    def __init__(self, datalist):
        """
        Object to check which plugins in datalist can share their spectrum calculation, because
        they have the same input energy bins and integration method. Can save a lot of time if the
        calculation of the spectrum is slow.
        """

        # List with different Ebin edges of the plugins
        self._data_ein_edges = []
        self._base_plugin_key = []
        # List with the information which plugins have the same spectrum integration
        # with same input energy bins
        self._data_ebin_connect = []
        #TODO add check if same integration method is set
        for j, (key, d) in enumerate(zip(list(datalist.keys()),
                                         list(datalist.values()))):
            if isinstance(d, DispersionSpectrumLike):
                e = d.response.monte_carlo_energies
                share_spec_possible = True
            elif isinstance(d, SpectrumLike):
                e = d.observed_spectrum.edges
                share_spec_possible = True
            else:
                log.debug(f"Plugin {j} can not share spectrum calculation (Not SpectrumLike or DispersionSpectrumLike)")
                self._data_ein_edges.append(
                    None
                )
                self._base_plugin_key.append(key)
                self._data_ebin_connect.append(j)
                share_spec_possible = False

            if share_spec_possible:
                found = False
                log.debug(f"Plugin {j} can share spectrum calculation")
                # Check if these Ein_bins are already used by an earlier plugin
                for i in range(len(self._data_ein_edges)):
                    if self._data_ein_edges[i] is not None:
                        if len(e) == len(self._data_ein_edges[i]):
                            if np.all(np.equal(e, self._data_ein_edges[i])):
                                log.debug(f"Plugin {j} shares the spectrum calculation with plugin {i}")
                                self._data_ebin_connect.append(i)
                                found = True
                                break
                # If not save these Ein_bins and add an entry to the connection array
                if not found:
                    self._data_ebin_connect.append(len(self._data_ein_edges))
                    self._data_ein_edges.append(e)
                    self._base_plugin_key.append(key)

    @property
    def data_ein_edges(self):
        return self._data_ein_edges

    @property
    def data_ebin_connect(self):
        return self._data_ebin_connect

    @property
    def base_plugin_key(self):
        return self._base_plugin_key
