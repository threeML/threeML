from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import object
import pandas as pd
import os
import yaml
import astropy.io.votable as votable
import astropy.units as u
import urllib.request, urllib.error, urllib.parse
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import numpy as np
import warnings

import speclite.filters as spec_filter

from threeML.io.configuration import get_user_data_path
from threeML.io.file_utils import (
    if_directory_not_existing_then_make,
    file_existing_and_readable,
)
from threeML.io.network import internet_connection_is_active
from threeML.io.package_data import get_path_of_data_dir


def get_speclite_filter_path():

    return os.path.join(get_path_of_data_dir(), "optical_filters")


def to_valid_python_name(name):

    new_name = name.replace("-", "_")

    try:

        int(new_name[0])

        new_name = "f_%s" % new_name

        return new_name

    except (ValueError):

        return new_name


class ObservatoryNode(object):
    def __init__(self, sub_dict):

        self._sub_dict = sub_dict

    def __repr__(self):
        return yaml.dump(self._sub_dict, default_flow_style=False)


class FilterLibrary(object):
    def __init__(self, library_file):
        """
        holds all the observatories/instruments/filters


        :param library_file:
        """

        # get the filter file

        with open(library_file) as f:

            self._library = yaml.safe_load(f)

        self._instruments = []

        # create attributes which are lib.observatory.instrument
        # and the instrument attributes are speclite FilterResponse objects

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print("Loading optical filters")

            for observatory, value in self._library.items():

                # create a node for the observatory
                this_node = ObservatoryNode(value)

                # attach it to the object

                setattr(self, observatory, this_node)

                # now get the instruments

                for instrument, value2 in value.items():

                    # update the instruments

                    self._instruments.append(instrument)

                    # create the filter response via speclite

                    filter_path = os.path.join(
                        get_speclite_filter_path(), observatory, instrument
                    )

                    filters_to_load = [
                        "%s-%s.ecsv" % (filter_path, filter) for filter in value2
                    ]

                    this_filter = spec_filter.load_filters(*filters_to_load)

                    # attach the filters to the observatory

                    setattr(this_node, instrument, this_filter)

        self._instruments.sort()

    @property
    def instruments(self):

        return self._instruments

    def __repr__(self):
        return yaml.dump(self._library, default_flow_style=False)


def add_svo_filter_to_speclite(observatory, instrument, ffilter, update=False):
    """
    download an SVO filter file and then add it to the user library
    :param observatory:
    :param instrument:
    :param ffilter:
    :return:
    """

    # make a directory for this observatory and instrument

    filter_path = os.path.join(
        get_speclite_filter_path(), to_valid_python_name(observatory)
    )

    if_directory_not_existing_then_make(filter_path)

    # grab the filter file from SVO

    # reconvert 2MASS so we can grab it

    if observatory == "TwoMASS":
        observatory = "2MASS"

    if (
        not file_existing_and_readable(
            os.path.join(
                filter_path,
                "%s-%s.ecsv"
                % (to_valid_python_name(instrument), to_valid_python_name(ffilter)),
            )
        )
        or update
    ):

        url_response = urllib.request.urlopen(
            "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?PhotCalID=%s/%s.%s/AB"
            % (observatory, instrument, ffilter)
        )
        # now parse it
        data = votable.parse_single_table(url_response).to_table()

        # save the waveunit

        waveunit = data["Wavelength"].unit

        # the filter files are masked arrays, which do not go to zero on
        # the boundaries. This confuses speclite and will throw an error.
        # so we add a zero on the boundaries

        if data["Transmission"][0] != 0.0:

            w1 = data["Wavelength"][0] * 0.9
            data.insert_row(0, [w1, 0])

        if data["Transmission"][-1] != 0.0:

            w2 = data["Wavelength"][-1] * 1.1
            data.add_row([w2, 0])

        # filter any negative values

        idx = data["Transmission"] < 0
        data["Transmission"][idx] = 0

        # build the transmission. # we will force all the wavelengths
        # to Angstroms because sometimes AA is misunderstood

        try:

            transmission = spec_filter.FilterResponse(
                wavelength=data["Wavelength"] * waveunit.to("Angstrom") * u.Angstrom,
                response=data["Transmission"],
                meta=dict(
                    group_name=to_valid_python_name(instrument),
                    band_name=to_valid_python_name(ffilter),
                ),
            )

            # save the filter

            transmission.save(filter_path)

            success = True

        except (ValueError):

            success = False

            print(
                "%s:%s:%s has an invalid wave table, SKIPPING"
                % (observatory, instrument, ffilter)
            )

        return success

    else:

        return True


def download_SVO_filters(filter_dict, update=False):
    """

    download the filters sets from the SVO repository


    :return:
    """

    # to group the observatory / instrument / filters

    search_name = re.compile("^(.*)\/(.*)\.(.*)$")

    # load the SVO meta XML file

    svo_url = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?"

    url_response = urllib.request.urlopen(svo_url)

    # the normal VO parser cannot read the XML table
    # so we manually do it to obtain all the instrument names

    tree = ET.parse(url_response)

    observatories = []

    for elem in tree.iter(tag="PARAM"):
        if elem.attrib["name"] == "INPUT:Facility":
            for child in list(elem):
                if child.tag == "VALUES":
                    for child2 in list(child):
                        val = child2.attrib["value"]

                        if val != "":

                            observatories.append(val)

    # now we are going to build a multi-layer dictionary
    # observatory:instrument:filter

    for obs in observatories:

        # fix 2MASS to a valid name

        if obs == "2MASS":

            obs = "TwoMASS"

        url_response = urllib.request.urlopen(
            "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?Facility=%s" % obs
        )

        try:

            # parse the VO table

            v = votable.parse(url_response)

            instrument_dict = defaultdict(list)

            # get the filter names for this observatory

            instruments = v.get_first_table().to_table()["filterID"].tolist()

            print("Downloading %s filters" % (obs))

            for x in instruments:

                _, instrument, subfilter = search_name.match(x).groups()

                success = add_svo_filter_to_speclite(obs, instrument, subfilter, update)

                if success:

                    instrument_dict[to_valid_python_name(instrument)].append(
                        to_valid_python_name(subfilter)
                    )

                    # attach this to the big dictionary

            filter_dict[to_valid_python_name(obs)] = dict(instrument_dict)

        except (IndexError):

            pass

    return filter_dict


def download_grond(filter_dict):

    save_path = os.path.join(get_speclite_filter_path(), "ESO")

    if_directory_not_existing_then_make(save_path)

    grond_filter_url = "http://www.mpe.mpg.de/~jcg/GROND/GROND_filtercurves.txt"

    url_response = urllib.request.urlopen(grond_filter_url)

    grond_table = pd.read_table(url_response)

    wave = grond_table["A"].as_matrix()

    bands = ["g", "r", "i", "z", "H", "J", "K"]

    for band in bands:

        curve = np.array(grond_table["%sBand" % band])
        curve[curve < 0] = 0
        curve[0] = 0
        curve[-1] = 0

        grond_spec = spec_filter.FilterResponse(
            wavelength=wave * u.nm,
            response=curve,
            meta=dict(group_name="GROND", band_name=band),
        )

        grond_spec.save(directory_name=save_path)

    filter_dict["ESO"] = {"GROND": bands}

    return filter_dict


def build_filter_library():

    if not file_existing_and_readable(
        os.path.join(get_speclite_filter_path(), "filter_lib.yml")
    ):

        print("Downloading optical filters. This will take a while.\n")

        if internet_connection_is_active():

            filter_dict = {}

            filter_dict = download_SVO_filters(filter_dict)

            filter_dict = download_grond(filter_dict)

            # Ok, finally, we want to keep track of the SVO filters we have
            # so we will save this to a YAML file for future reference
            with open(
                os.path.join(get_speclite_filter_path(), "filter_lib.yml"), "w"
            ) as f:

                yaml.safe_dump(filter_dict, f, default_flow_style=False)

            return True

        else:

            print(
                "You do not have the 3ML filter library and you do not have an active internet connection."
            )
            print("Please connect to the internet to use the 3ML filter library.")
            print("pyspeclite filter library is still available.")

            return False

    else:

        return True


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    lib_exists = build_filter_library()

if lib_exists:

    threeML_filter_library = FilterLibrary(
        os.path.join(get_speclite_filter_path(), "filter_lib.yml")
    )

    __all__ = ["threeML_filter_library"]

else:

    raise RuntimeError("The threeML filter library does not exist!")
