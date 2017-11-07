from threeML.io.file_utils import sanitize_filename, if_directory_not_existing_then_make, file_existing_and_readable
from threeML.config.config import threeML_config
from threeML.io.download_from_http import ApacheDirectory, RemoteDirectoryNotFound
from threeML.io.dict_with_pretty_print import DictWithPrettyPrint

from threeML.exceptions.custom_exceptions import TriggerDoesNotExist

import gzip
import shutil
import os
import numpy as np
from collections import OrderedDict
import re


def _validate_fermi_trigger_name(trigger):

    _trigger_name_match = re.compile("^(bn|grb?)? ?(\d{9})$")

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

    return trigger


_month2number = {'jan':1,
                  'january':1,
                  'feb':2,
                  'february':2,
                  'mar':3,
                  'march':3,
                  'apr':4,
                  'april':4,
                  'may':5,
                  'jun':6,
                  'june':6,
                  'jul':7,
                  'july':7,
                  'aug':8,
                  'august':8,
                  'sept':9,
                  'september':9,
                  'oct':10,
                  'october':10,
                  'nov':11,
                  'november':11,
                  'dec':12,
                 'december':12}

def _validate_fermi_daily_name(year,month,day):


    # check the year



    assert type(year) == int, 'Invalid year argument'

    assert year >= 2008, '%d is not a valid year'%year




    if type(month) == int:

        assert month in range(1,13), '%d is not a valid month' %month



    elif type(month) == str:

        assert (month.lower() in _month2number), '%s is not a valid month. use %s' % (month,', '.join(_month2number.keys()))

        month = _month2number[month]

    else:

        RuntimeError('Invalid month argument')


    month_str = ("%d" % month).zfill(2)


    assert type(day) == int, 'Invalid day argument'

    assert day in range(1,32), '%d is not a valid day' %day

    day_str = ("%d" % day).zfill(2)


    full_day_string = '%d/%s/%s/current' % (year,month_str,day_str)


    return full_day_string


_detector_list = 'n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,na,nb,b0,b1'.split(",")


def download_GBM_trigger_data(trigger_name, detectors=None, destination_directory='.', compress_tte=True):
    """
    Download the latest GBM TTE and RSP files from the HEASARC server. Will get the
    latest file version and prefer RSP2s over RSPs. If the files already exist in your destination
    directory, they will be skipped in the download process. The output dictionary can be used
    as input to the FermiGBMTTELike class.

    example usage: download_GBM_trigger_data('080916009', detectors=['n0','na','b0'], destination_directory='.')

    :param trigger_name: trigger number (str) e.g. '080916009' or 'bn080916009' or 'GRB080916009'
    :param detectors: list of detectors, default is all detectors
    :param destination_directory: download directory
    :param compress_tte: compress the TTE files via gzip (default True)
    :return: a dictionary with information about the download
    """

    # Let's doctor up the input just in case the user tried something strange

    sanitized_trigger_name_ = _validate_fermi_trigger_name(trigger_name)

    # create output directory if it does not exists
    destination_directory = sanitize_filename(destination_directory, abspath=True)

    if_directory_not_existing_then_make(destination_directory)

    # Sanitize detector list (if any)
    if detectors is not None:

        for det in detectors:

            assert det in _detector_list, "Detector %s in the provided list is not a valid detector. " \
                                          "Valid choices are: %s" % (det, _detector_list)

    else:

        detectors = list(_detector_list)

    # Open heasarc web page

    url = threeML_config['gbm']['public HTTP location']
    year = '20%s' % sanitized_trigger_name_[:2]
    directory = '/triggers/%s/bn%s/current' % (year, sanitized_trigger_name_)

    heasarc_web_page_url = '%s/%s' % (url, directory)

    try:

        downloader = ApacheDirectory(heasarc_web_page_url)

    except RemoteDirectoryNotFound:

        raise TriggerDoesNotExist("Trigger %s does not exist at %s" % (sanitized_trigger_name_, heasarc_web_page_url))

    # Now select the files we want to download, then we will download them later
    # We do it in two steps because we want to be able to choose what to download once we
    # have the complete picture

    # Get the list of remote files
    remote_file_list = downloader.files

    # This is the dictionary to keep track of the classification
    remote_files_info = DictWithPrettyPrint([(det, {}) for det in detectors])

    # Classify the files detector by detector

    for this_file in remote_file_list:

        # this_file is something like glg_tte_n9_bn100101988_v00.fit
        tokens = this_file.split("_")

        if len(tokens) != 5:

            # Not a data file

            continue

        else:

            # The "map" is necessary to transform the tokens to normal string (instead of unicode),
            # because u"b0" != "b0" as a key for a dictionary

            _, file_type, detname, _, version_ext = map(str, tokens)

        version, ext = version_ext.split(".")

        # We do not care here about the other files (tcat, bcat and so on),
        # nor about files which pertain to other detectors

        if file_type not in ['cspec', 'tte'] or ext not in ['rsp','rsp2','pha','fit'] or detname not in detectors:

            continue

        # cspec files can be rsp, rsp2 or pha files. Classify them

        if file_type == 'cspec':

            if ext == 'rsp':

                remote_files_info[detname]['rsp'] = this_file

            elif ext == 'rsp2':

                remote_files_info[detname]['rsp2'] = this_file

            elif ext == 'pha':

                remote_files_info[detname]['cspec'] = this_file

            else:

                raise RuntimeError("Should never get here")

        else:

            remote_files_info[detname][file_type] = this_file

    # Now download the files

    download_info = DictWithPrettyPrint([(det, DictWithPrettyPrint()) for det in detectors])

    for detector in remote_files_info.keys():

        remote_detector_info = remote_files_info[detector]
        local_detector_info = download_info[detector]

        # Get CSPEC file
        local_detector_info['cspec'] = downloader.download(remote_detector_info['cspec'], destination_directory,
                                                           progress=True)

        # Get the RSP2 file if it exists, otherwise get the RSP file
        if 'rsp2' in remote_detector_info:

            local_detector_info['rsp'] = downloader.download(remote_detector_info['rsp2'], destination_directory,
                                                              progress=True)

        else:

            local_detector_info['rsp'] = downloader.download(remote_detector_info['rsp'], destination_directory,
                                                             progress=True)

        # Get TTE file (compressing it if requested)
        local_detector_info['tte'] = downloader.download(remote_detector_info['tte'], destination_directory,
                                                         progress=True, compress=compress_tte)

    return download_info


def _get_latest_version(filenames):
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


def cleanup_downloaded_GBM_data(detector_information_dict):
    """
    deletes data downloaded with download_GBM_trigger_data.
    :param detector_information_dict: the return dictionary from download_GBM_trigger_data
    """
    # go through each detector
    for detector in detector_information_dict.keys():

        # for each detector, remove the data file
        for data_file in detector_information_dict[detector].values():
            print("Removing: %s" % data_file)

            os.remove(data_file)

    print('\n')


def download_GBM_daily_data(year,month,day, detectors=None, data_type=None,destination_directory='.', compress_tte=True):
    """
    Download the latest continuous GBM TTE and CSPEC for the specified day from theHEASARC server.  If the files already exist in your destination
    directory, they will be skipped in the download process. The output dictionary can be used
    as input to the FermiGBMTTELike class.

    example usage: download_GBM_daily_data(2009,1,12, detectors=['n0','na','b0'], destination_directory='.')

    :param year: the year of the data
    :month: the month of the data eithera number 1-12 or an month 'feb' or 'february'
    :param day: day of the month
    :param data_type: None= all or tte or cspec
    :param destination_directory: download directory
    :param compress_tte: compress the TTE files via gzip (default True)
    :return: a dictionary with information about the download
    """

    # Let's doctor up the input just in case the user tried something strange

    sanitized_trigger_name = _validate_fermi_daily_name(year,month,day)

    # create output directory if it does not exists
    destination_directory = sanitize_filename(destination_directory, abspath=True)

    if_directory_not_existing_then_make(destination_directory)

    # Sanitize detector list (if any)
    if detectors is not None:

        for det in detectors:

            assert det in _detector_list, "Detector %s in the provided list is not a valid detector. " \
                                          "Valid choices are: %s" % (det, _detector_list)

    else:

        detectors = list(_detector_list)

    # Open heasarc web page

    url = threeML_config['gbm']['public HTTP location']

    directory = 'daily/%s'% sanitized_trigger_name

    heasarc_web_page_url = '%s/%s' % (url, directory)

    try:

        downloader = ApacheDirectory(heasarc_web_page_url)

    except RemoteDirectoryNotFound:

        raise TriggerDoesNotExist("Day %s does not exist at %s" % (sanitized_trigger_name, heasarc_web_page_url))

    # Now select the files we want to download, then we will download them later
    # We do it in two steps because we want to be able to choose what to download once we
    # have the complete picture

    # Get the list of remote files
    remote_file_list = downloader.files

    # This is the dictionary to keep track of the classification
    remote_files_info = DictWithPrettyPrint([(det, {}) for det in detectors])

    # Classify the files detector by detector

    tte_info = {}

    for this_file in remote_file_list:

        # this_file is something like glg_tte_n9_bn100101988_v00.fit
        tokens = this_file.split("_")

        if len(tokens) != 5:

            # Not a data file

            continue

        else:

            # The "map" is necessary to transform the tokens to normal string (instead of unicode),
            # because u"b0" != "b0" as a key for a dictionary

            _, file_type, detname, time_tag, version_ext = map(str, tokens)


        try:
            version, ext = version_ext.split(".")

        except(ValueError):

            # sometimes GBM randomly GZips the data... so we have to deal that

            version,ext,fmt = version_ext.split('.')

            compress_tte = False


        # We do not care here about the other files (posthist, spechist and so on),
        # nor about files which pertain to other detectors

        if file_type not in ['cspec', 'tte'] or ext not in ['pha','fit'] or detname not in detectors:

            continue

        else:

            if file_type == 'tte':

                # the TTE files are broken into slices
                # so we must account for that

                if remote_files_info[detname][file_type] is None:

                    remote_files_info[detname][file_type] = {}

                remote_files_info[detname][file_type][time_tag] = this_file




            else:

                remote_files_info[detname][file_type] = this_file

    # Now download the files

    download_info = DictWithPrettyPrint([(det, DictWithPrettyPrint()) for det in detectors])

    _warned_tte = False



    for detector in remote_files_info.keys():



        remote_detector_info = remote_files_info[detector]
        local_detector_info = download_info[detector]

        # Get CSPEC file

        if (data_type is None) or (data_type == 'cspec'):

            local_detector_info['cspec'] = downloader.download(remote_detector_info['cspec'], destination_directory,
                                                               progress=True)

        # Get TTE file (compressing it if requested)
        if (data_type is None) or (data_type=='tte') and ('tte' in local_detector_info):

            for time_tag in remote_detector_info['tte'].keys():

                local_detector_info['tte'] = downloader.download(time_tag, destination_directory,
                                                                 progress=True, compress=compress_tte)

        else:
            if not _warned_tte or data_type=='tte':

                print('Continuous TTE data does not exist for this day')

                _warned_tte = True

    return download_info


def _get_latest_version(filenames):
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


