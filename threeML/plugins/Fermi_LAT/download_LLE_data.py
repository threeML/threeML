from threeML.io.file_utils import sanitize_filename, file_existing_and_readable
from threeML.config.config import threeML_config
from threeML.io.download_from_ftp import download_files_from_directory_ftp

import ftplib
import glob
import os
import numpy as np
from collections import OrderedDict


class InvalidTrigger(RuntimeError):
    pass


class TriggerDoesNotExist(RuntimeError):
    pass


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

    _valid_trigger_args = ['080916008', 'bn080916009', 'GRB080916009']

    assert type(trigger) == str, "The trigger argument must be a string. Must be in the form %s" % (
        ', or '.join(_valid_trigger_args))

    # if there is the 'bn' on the front:
    test = trigger.lower().split('bn')
    # if they did, we will grab the proper part
    if len(test) == 2:

        trigger = test[-1]

    # if there is the 'GRB' on the front:
    test = trigger.lower().split('grb')
    # if they did, we will grab the proper part
    if len(test) == 2:

        trigger = test[-1]

    assert len(trigger) == 9, "The trigger argument is not valid. Must be in the form %s" % (
        ', or '.join(_valid_trigger_args))

    for trial in trigger:

        try:

            int(trial)

        except(ValueError):

            raise InvalidTrigger(
                    "The trigger argument is not valid. Must be in the form %s" % (', or '.join(_valid_trigger_args)))

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



    # now see if we have already downloaded these files
    lle_filter = np.ones_like(lle_to_get_latest, dtype=bool)
    rsp_filter = np.ones_like(rsp_to_get_latest, dtype=bool)
    ft2_filter = np.ones_like(ft2_to_get_latest, dtype=bool)


    for i, rsp in enumerate(rsp_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, rsp)):

            rsp_filter[i] = 0

            print('%s already downloaded into %s -> skipping' % (rsp, destination_directory))

    for i, lle in enumerate(lle_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, lle)):

            lle_filter[i] = 0

            print('%s already downloaded into %s -> skipping' % (lle, destination_directory))

    for i, ft2 in enumerate(ft2_to_get_latest):

        if file_existing_and_readable(os.path.join(destination_directory, ft2)):

            ft2_filter[i] = 0

            print('%s already downloaded into %s -> skipping' % (ft2, destination_directory))

    # now download the files

    remote_path = "%s/%s/" % (threeML_config['LAT']['public FTP location'], directory)

    retrieval = np.append(rsp_to_get_latest[rsp_filter], lle_to_get_latest[lle_filter])
    retrieval = np.append(retrieval, ft2_to_get_latest[ft2_filter])

    if len(retrieval) > 0:

        print("\nDownloading LLE, RSP and FT2 files...")

        downloaded_files = download_files_from_directory_ftp(remote_path,
                                                             sanitize_filename(destination_directory),
                                                             filenames=retrieval)

        rsp_files = downloaded_files[:len(rsp_to_get_latest[rsp_filter])]
        ft2_files = downloaded_files[len(rsp_to_get_latest[rsp_filter]):len(ft2_to_get_latest[ft2_filter])]
        lle_files = downloaded_files[len(lle_to_get_latest[lle_filter]):]

        print rsp_files
        print ft2_files
        print lle_files

    else:

        rsp_files = []
        lle_files = []
        ft2_files = []

    download_info = {}

    if rsp_files:

        download_info['rsp'] = rsp_files[0]

    else:

        download_info['rsp'] = os.path.join(destination_directory, rsp_to_get_latest[~rsp_filter][0])

    if lle_files:

        download_info['lle'] = lle_files[0]

    else:

        download_info['lle'] = os.path.join(destination_directory, lle_to_get_latest[~lle_filter][0])

    if ft2_files:

        download_info['ft2'] = ft2_files[0]

    else:

        download_info['ft2'] = os.path.join(destination_directory, ft2_to_get_latest[~ft2_filter][0])

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
