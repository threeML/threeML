from threeML.io.file_utils import sanitize_filename, file_existing_and_readable
from threeML.config.config import threeML_config
from threeML.io.download_from_ftp import download_files_from_directory_ftp
from threeML.exceptions.custom_exceptions import TriggerDoesNotExist

import ftplib
import re
import os
import numpy as np
from collections import OrderedDict



_trigger_name_match=re.compile("^(bn|grb?)? ?(\d{9})$")
_file_type_match = re.compile('gll_(\D{2,5})_bn\d{9}_v\d{2}\.\D{3}')
_valid_file_type = ['cspec','pt','lle']

def download_LLE_trigger_data(trigger, destination_directory='.'):
    """
    Download the latest Fermi LAT LLE and RSP files from the HEASARC server. Will get the
    latest file versions. If the files already exist in your destination
    directory, they will be skipped in the download process. The output dictionary can be used
    as input to the FermiLATLLELike class.

    example usage: download_LLE_trigger_data('080916009', destination_directory='.')

    :param trigger: trigger number (str) with no leading letter e.g. '080916009'
    :param destination_directory: download directory
    :return: a dictionary with information about the download
    """

    _valid_trigger_args = ['080916009', 'bn080916009', 'GRB080916009']

    assert_string = "The trigger %s is not valid. Must be in the form %s" % (trigger,
                                                                             ', or '.join(
                                                                                 _valid_trigger_args))

    assert type(trigger) == str, "triggers must be strings"

    trigger = trigger.lower()

    search = _trigger_name_match.match(trigger)

    assert search is not None, assert_string

    assert search.group(2) is not None, assert_string

    trigger = search.group(2)

    # create output directory if it does not exists
    destination_directory = sanitize_filename(destination_directory, abspath=True)

    if not os.path.exists(destination_directory):

        os.makedirs(destination_directory)

    # open and FTP to look at the data
    ftp = ftplib.FTP('legacy.gsfc.nasa.gov', 'anonymous', '')

    year = '20%s' % trigger[:2]
    directory = 'lat/triggers/%s/bn%s/current' % (year, trigger)

    directory_ = 'fermi/data/%s' % directory

    try:
        ftp.cwd(directory_)
    except ftplib.error_perm:

        ftp.quit()
        raise TriggerDoesNotExist("Trigger %s does not exist at the FSSC." % trigger)

    file_list = ftp.nlst()

    # kill this quick or urllib will get confused
    ftp.quit()

    del ftp

    # collect the rsp and tte files from the ftp list

    rsp_to_get = []

    for filename in file_list:
        if filename.find("gll_cspec") >= 0 and filename.find(".rsp") >= 0:
            rsp_to_get.append(filename)

    lle_to_get = []

    for filename in file_list:
        if filename.find("gll_lle") >= 0 and filename.find('.fit') >= 0:
            lle_to_get.append(filename)

    ft2_to_get = []

    for filename in file_list:
        if filename.find("gll_pt") >= 0 and filename.find('.fit') >= 0:
            ft2_to_get.append(filename)

    # lets make sure we get the latest versions of the files
    # prefer RSP2s

    rsp_to_get_latest = np.array(_get_latest_verison(rsp_to_get))

    lle_to_get_latest = np.array(_get_latest_verison(lle_to_get))

    ft2_to_get_latest = np.array(_get_latest_verison(ft2_to_get))



    files_to_download =[]
    files_existing = []


    for i, rsp in enumerate(rsp_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, rsp)):

            files_existing.append(rsp)

            print('%s already downloaded into %s -> skipping' % (rsp, destination_directory))

        else:

            files_to_download.append(rsp)

    for i, lle in enumerate(lle_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, lle)):

            files_existing.append(lle)

            print('%s already downloaded into %s -> skipping' % (lle, destination_directory))

        else:

            files_to_download.append(lle)

    for i, ft2 in enumerate(ft2_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, ft2)):

            files_existing.append(ft2)

            print('%s already downloaded into %s -> skipping' % (ft2, destination_directory))

        else:

            files_to_download.append(ft2)

    # now download the files

    remote_path = "%s/%s/" % (threeML_config['LAT']['public FTP location'], directory)

    download_info = {}

    file_lookup = {'lle':'lle','pt':'ft2','cspec':'rsp'}

    if len(files_to_download) > 0:

        print("\nDownloading LLE, RSP and FT2 files...")

        downloaded_files = download_files_from_directory_ftp(remote_path,
                                                             sanitize_filename(destination_directory),
                                                             filenames=files_to_download)

        # rsp_files = downloaded_files[:len(rsp_to_get_latest[rsp_filter])]
        # lle_files = downloaded_files[len(rsp_to_get_latest[rsp_filter]):len(rsp_to_get_latest[rsp_filter]) + len(
        #         ft2_to_get_latest[ft2_filter])]
        # ft2_files = downloaded_files[len(rsp_to_get_latest[rsp_filter]) + len(ft2_to_get_latest[ft2_filter]):]




        for download in downloaded_files:

            file_type = _file_type_match.match(download.split("/")[-1]).group(1)

            assert file_type in _valid_file_type, "Something went wrong %s is not an LLE, RSP, or FT2 file" % download.split("/")[-1] #pragma: no cover

            download_info[file_lookup[file_type]] = download









    for downloaded in files_existing:

        file_type = _file_type_match.match(downloaded).group(1)

        assert file_type in _valid_file_type, "Something went wrong %s is not an LLE, RSP, or FT2 file" % download # pragma: no cover

        download_info[file_lookup[file_type]]= os.path.join(destination_directory, downloaded)







    return download_info


def _get_latest_verison(filenames):
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
        fn_stub, vn_stub = fn.split('_v')

        # split the vn string and extension
        vn_string, ext = vn_stub.split('.')

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

    for key in vn_as_num.keys():

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

    for data_file in detector_information_dict.values():

        print("Removing: %s"%data_file)

        os.remove(data_file)

    print('\n')