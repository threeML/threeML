from threeML.io.file_utils import sanitize_filename
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


def download_GBM_trigger_data(trigger, detectors=None, destination_directory='.'):
    """
    Download the latest GBM TTE and RSP files from the HEASARC server. Will get the
    latest file version and prefer RSP2s over RSPs. If the files already exist in your destination
    directory, they will be skipped in the download process. The output dictionary can be used
    as input to the FermiGBMTTELike class.

    example usage: download_GBM_trigger_data('080916009', detectors=['n0','na','b0'], destination_directory='.')

    :param trigger: trigger number (str) e.g. '080916009' or 'bn080916009' or 'GRB080916009'
    :param detectors: list of detectors, default is all detectors
    :param destination_directory: download directory
    :return: a dictionary with information about the download
    """

    # Let's doctor up the input just in case the user tried something strange

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
    directory = 'gbm/triggers/%s/bn%s/current' % (year, trigger)

    allowed_detectors = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

    if detectors is None:

        detectors = allowed_detectors

    else:

        assert type(detectors) == list, 'detectors must be input as a list'

        for detector in detectors:

            assert detector in allowed_detectors, '%s is not a valid GBM detector. Allowed choices %s' % (
                detector, ' ,'.join(allowed_detectors))

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
        for det in detectors:
            if filename.find(".rsp") >= 0 and filename.find("cspec") >= 0 and filename.find(det + '_') >= 0:
                rsp_to_get.append(filename)

    tte_to_get = []

    for filename in file_list:
        for det in detectors:
            if filename.find("tte") >= 0 and filename.find(det + '_') >= 0:
                tte_to_get.append(filename)

    # lets make sure we get the latest versions of the files
    # prefer RSP2s

    rsp_to_get_latest = _get_latest_verison(rsp_to_get)

    tte_to_get_latest = _get_latest_verison(tte_to_get)

    assert len(tte_to_get_latest) == len(
            rsp_to_get_latest), 'The file list should be the same length. Something went wrong. Contact grburgess'

    # reorder the file names to match the detectors
    # the dictionary keys cause them to get out of order

    tmp_rsp = []
    tmp_tte = []

    for det in detectors:

        for rsp in rsp_to_get_latest:

            if rsp.find('_%s_' % det) >= 0:

                tmp_rsp.append(rsp)

        for tte in tte_to_get_latest:

            if tte.find('_%s_' % det) >= 0:

                tmp_tte.append(tte)

    tte_to_get_latest = np.array(tmp_tte)
    rsp_to_get_latest = np.array(tmp_rsp)

    # now see if we have already downloaded these files
    tte_filter = np.ones_like(tte_to_get_latest, dtype=bool)
    rsp_filter = np.ones_like(rsp_to_get_latest, dtype=bool)

    for i, rsp in enumerate(rsp_to_get_latest):

        test = glob.glob(os.path.join(destination_directory, rsp))

        if test:

            rsp_filter[i] = 0

            print('%s already downloaded into %s -> skipping' % (rsp, destination_directory))

    for i, tte in enumerate(tte_to_get_latest):

        test = glob.glob(os.path.join(destination_directory, tte))

        if test:

            tte_filter[i] = 0

            print('%s already downloaded into %s -> skipping' % (tte, destination_directory))

    # now download the files

    remote_path = "%s/%s/" % (threeML_config['LAT']['public FTP location'], directory)

    retrival = np.append(rsp_to_get_latest[rsp_filter], tte_to_get_latest[tte_filter])

    if len(retrival) > 0:

        print("\nDownloading TTE and RSP files...")

        downloaded_files = download_files_from_directory_ftp(remote_path,
                                                             sanitize_filename(destination_directory),
                                                             filenames=retrival)

        rsp_files = downloaded_files[:len(rsp_to_get_latest[rsp_filter])]
        tte_files = downloaded_files[len(tte_to_get_latest[tte_filter]):]

    else:

        rsp_files = []
        tte_files = []

    download_info = {}

    detectors = np.array(detectors)

    for detector in detectors:

        download_info[detector] = {}

    # firrt handle the rsp/tte from the DL'ed detectors
    for detector, rsp in zip(detectors[rsp_filter], rsp_files):

        download_info[detector]['rsp'] = rsp

    for detector, tte in zip(detectors[tte_filter], tte_files):

        download_info[detector]['tte'] = tte

    # now handle the skipped files

    for detector, rsp in zip(detectors[~rsp_filter], rsp_to_get_latest[~rsp_filter]):

        download_info[detector]['rsp'] = os.path.join(destination_directory, rsp)

    for detector, tte in zip(detectors[~tte_filter], tte_to_get_latest[~tte_filter]):

        download_info[detector]['tte'] = os.path.join(destination_directory, tte)

    return download_info


def _get_latest_verison(filenames):
    """
    returns the list with only the highest version numbers selected

    :param filenames: list of GBM data files
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

        # first we favor RSP2

        ext = np.array(extentions[key])

        idx = ext == 'rsp2'

        # if there are no rsp2 in the files
        if idx.sum() == 0:

            # we can select on all
            idx = np.ones_like(ext, dtype=bool)

        ext = ext[idx]
        vn = np.array(vn_as_num[key])[idx]
        vn_string = np.array(vn_as_string[key])[idx]

        # get the latest version
        max_vn = np.argmax(vn)

        # create the file name
        latest_version = "%s_v%s.%s" % (key, vn_string[max_vn], ext[max_vn])

        final_file_names.append(latest_version)

    return final_file_names
