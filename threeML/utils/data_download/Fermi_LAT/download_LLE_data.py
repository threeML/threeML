from __future__ import print_function
from threeML.io.file_utils import sanitize_filename, if_directory_not_existing_then_make
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import TriggerDoesNotExist
from threeML.io.download_from_http import ApacheDirectory, RemoteDirectoryNotFound
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint
from threeML.utils.data_download.Fermi_GBM.download_GBM_data import (
    _validate_fermi_trigger_name,
)

import re
import os
import numpy as np
from collections import OrderedDict


_trigger_name_match = re.compile("^(bn|grb?)? ?(\d{9})$")
_file_type_match = re.compile("gll_(\D{2,5})_bn\d{9}_v\d{2}\.\D{3}")


def download_LLE_trigger_data(trigger_name, destination_directory="."):
    """
    Download the latest Fermi LAT LLE and RSP files from the HEASARC server. Will get the
    latest file versions. If the files already exist in your destination
    directory, they will be skipped in the download process. The output dictionary can be used
    as input to the FermiLATLLELike class.

    example usage: download_LLE_trigger_data('080916009', destination_directory='.')

    :param trigger_name: trigger number (str) with no leading letter e.g. '080916009'
    :param destination_directory: download directory
    :return: a dictionary with information about the download
    """

    sanitized_trigger_name_ = _validate_fermi_trigger_name(trigger_name)

    # create output directory if it does not exists
    destination_directory = sanitize_filename(destination_directory, abspath=True)
    if_directory_not_existing_then_make(destination_directory)

    # Figure out the directory on the server
    url = threeML_config["LAT"]["public HTTP location"]

    year = "20%s" % sanitized_trigger_name_[:2]
    directory = "triggers/%s/bn%s/current" % (year, sanitized_trigger_name_)

    heasarc_web_page_url = "%s/%s" % (url, directory)

    try:

        downloader = ApacheDirectory(heasarc_web_page_url)

    except RemoteDirectoryNotFound:

        raise TriggerDoesNotExist(
            "Trigger %s does not exist at %s"
            % (sanitized_trigger_name_, heasarc_web_page_url)
        )

    # Download only the lle, pt, cspec and rsp file (i.e., do not get all the png, pdf and so on)
    pattern = "gll_(lle|pt|cspec)_bn.+\.(fit|rsp|pha)"

    destination_directory_sanitized = sanitize_filename(destination_directory)

    downloaded_files = downloader.download_all_files(
        destination_directory_sanitized, progress=True, pattern=pattern
    )

    # Put the files in a structured dictionary

    download_info = DictWithPrettyPrint()

    for download in downloaded_files:

        file_type = _file_type_match.match(os.path.basename(download)).group(1)

        if file_type == "cspec":

            # a cspec file can be 2 things: a CSPEC spectral set (with .pha) extension,
            # or a response matrix (with a .rsp extension)

            ext = os.path.splitext(os.path.basename(download))[1]

            if ext == ".rsp":

                file_type = "rsp"

            elif ext == ".pha":

                file_type = "cspec"

            else:

                raise RuntimeError("Should never get here")

        # The pt file is really an ft2 file

        if file_type == "pt":

            file_type = "ft2"

        download_info[file_type] = download

    return download_info


def _get_latest_version(filenames):
    """
    returns the list with only the highest version numbers selected

    :param filenames: list of LLE data files
    :return:
    """

    # this holds the version number
    vn_as_num = OrderedDict()

    # this holds the extensions
    extentions = OrderedDict()

    # this holds the vn string
    vn_as_string = OrderedDict()

    for fn in filenames:

        # get the first part of the file
        fn_stub, vn_stub = fn.split("_v")

        # split the vn string and extension
        vn_string, ext = vn_stub.split(".")

        # convert the vn to a number
        vn = 0
        for i in vn_string:
            vn += int(i)

        # build the dictionaries where keys
        # are the non-unique file name
        # and values are extensions and vn

        vn_as_num.setdefault(fn_stub, []).append(vn)
        extentions.setdefault(fn_stub, []).append(ext)
        vn_as_string.setdefault(fn_stub, []).append(vn_string)

    final_file_names = []

    # Now we we go through and make selections

    for key in list(vn_as_num.keys()):

        ext = np.array(extentions[key])
        vn = np.array(vn_as_num[key])
        vn_string = np.array(vn_as_string[key])

        # get the latest version
        max_vn = np.argmax(vn)

        # create the file name
        latest_version = "%s_v%s.%s" % (key, vn_string[max_vn], ext[max_vn])

        final_file_names.append(latest_version)

    return final_file_names


def cleanup_downloaded_LLE_data(detector_information_dict):
    """
    deletes data downloaded with download_LLE_trigger_data.
    :param detector_information_dict: the return dictionary from download_LLE_trigger_data
    """

    for data_file in list(detector_information_dict.values()):

        print("Removing: %s" % data_file)

        os.remove(data_file)

    print("\n")
