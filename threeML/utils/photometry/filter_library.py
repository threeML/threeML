
from pathlib import Path

import astropy.units as u
import h5py
import speclite.filters as spec_filter
import yaml
from threeML.utils.progress_bar import tqdm

from threeML.io.package_data import get_path_of_data_dir


def get_speclite_filter_path() -> Path:

    return get_path_of_data_dir() / "optical_filters"


def get_speclite_filter_library() -> Path:

    return get_speclite_filter_path() / "filter_library.h5"


class ObservatoryNode(object):
    def __init__(self, sub_dict):

        self._sub_dict = sub_dict

    def __repr__(self):
        return yaml.dump(self._sub_dict, default_flow_style=False)


class FilterLibrary(object):
    def __init__(self):
        """
        holds all the observatories/instruments/filters


        :param library_file:
        """

        # get the filter file

        with h5py.File(get_speclite_filter_library(), "r") as f:

            self._instruments = []

            for observatory in tqdm(f.keys(), desc="Loading photometric filters"):

                sub_dict = {}
                for instrument in f[observatory].keys():

                    sub_dict[instrument] = instrument

                # create a node for the observatory
                this_node = ObservatoryNode(sub_dict)

                # attach it to the object

                if observatory == "2MASS":

                    xx = "TwoMass"

                else:

                    xx = observatory

                setattr(self, xx, this_node)

                # now get the instruments

                for instrument in f[observatory].keys():

                    # update the instruments

                    self._instruments.append(instrument)

                    # create the filter response via speclite

                    this_grp = f[observatory][instrument]
                    filters = []

                    for ff in this_grp.keys():

                        grp = this_grp[ff]

                        this_filter = spec_filter.FilterResponse(
                            wavelength=grp["wavelength"][()] * u.Angstrom,
                            response=grp["transmission"][()],
                            meta=dict(
                                group_name=instrument,
                                band_name=ff,
                            )
                        )

                        filters.append(this_filter)

                    fgroup = spec_filter.FilterSequence(filters)
                    # attach the filters to the observatory

                    setattr(this_node, instrument, fgroup)

        self._instruments.sort()

    @property
    def instruments(self):

        return self._instruments

    # def __repr__(self):
    #     return yaml.dump(self._library, default_flow_style=False)


def get_photometric_filter_library():
    """
    Get the 3ML filter library
    """
    if get_speclite_filter_library().exists():

        return FilterLibrary()

    else:

        raise RuntimeError("The threeML filter library does not exist!")
