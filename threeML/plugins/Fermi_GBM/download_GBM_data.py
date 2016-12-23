import HTMLParser
import html2text
import re
import socket
import time
import urllib
import os
import glob

import astropy.io.fits as pyfits

from threeML.io.file_utils import sanitize_filename
from threeML.config.config import threeML_config
from threeML.io.download_from_ftp import download_files_from_directory_ftp
from threeML.utils.unique_deterministic_tag import get_unique_deterministic_tag

from __future__ import print_function
import ftplib
import sys, time
import datetime


class GetGBMData(object):
    def __init__(self, trigger=None, daily=None):

        self.ftp = ftplib.FTP('legacy.gsfc.nasa.gov', 'anonymous', 'crap@gmail.com')

        if trigger is not None:

            self._type = 'triggered'

            year = '20' + trigger[:2]
            self._directory = 'fermi/data/gbm/triggers/' + year + '/bn' + trigger + '/current'

        elif daily is not None:

            self._type = 'daily'

            date = daily.split('/')

            self._directory = 'fermi/data/gbm/daily/20' + date[2] + '/' + date[0] + '/' + date[1] + '/current'


        else:

            print("You did something wrong!")
            return

        try:
            self.ftp.cwd(self._directory)
        except ftplib.error_perm:
            print(self._directory)
            print("Awww snap! This data entry does not exist at the FSSC. Exiting!\n")
            self.ftp.quit()
            return

        self._file_list = self.ftp.nlst()

        self._detectors = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

        self._where = ''

    def select_detectors(self, *dets):
        self._detectors = dets

        self._num_dets = len(dets)

    def set_destination(self, destination):

        self._where = destination

    def _get(self, items):

        # n_items = len(items)
        #
        # progress = ProgressBar(n_items)
        #
        # progress_bar_iter = max(int(n_items / 100), 1)

        for i, item in enumerate(items):

            if i % progress_bar_iter == 0:
                progress.animate((i + 1))

            self.ftp.retrbinary('RETR ' + item, open(self._where + item, 'wb').write)

        progress.animate(n_items)

    def get_rsp_cspec(self):

        to_get = []

        for i in self._file_list:
            for j in self._detectors:
                if ".rsp" in i and "cspec" in i and j + '_' in i:
                    to_get.append(i)
        self._get(to_get)

    def get_rsp_ctime(self):

        to_get = []

        for i in self._file_list:
            for j in self._detectors:
                if ".rsp" in i and "ctime" in i and j + '_' in i:
                    to_get.append(i)
        self._get(to_get)

    def get_ctime(self):

        to_get = []

        for i in self._file_list:
            for j in self._detectors:
                if "ctime" in i and j + '_' in i and 'rsp' not in i:
                    to_get.append(i)
        self._get(to_get)

    def get_cspec(self):

        to_get = []

        for i in self._file_list:
            for j in self._detectors:
                if "cspec" in i and j + '_' in i and 'rsp' not in i:
                    to_get.append(i)

        self._get(to_get)

    def get_trigdat(self):

        to_get = []

        for i in self._file_list:
            if "trigdat" in i:
                to_get.append(i)

        self._get(to_get)

    def get_tte(self):

        to_get = []

        for i in self._file_list:
            for j in self._detectors:
                if "tte" in i and j + '_' in i:
                    to_get.append(i)

        self._get(to_get)

    def get_tcat(self):

        to_get = []

        for i in self._file_list:
            if "tcat" in i:
                to_get.append(i)

        self._get(to_get)

    def get_plots(self):

        to_get = []

        for i in self._file_list:
            if ".gif" in i or ".pdf" in i:
                to_get.append(i)

        self._get(to_get)

    def get_poshist(self):

        to_get = []

        for i in self._file_list:
            if "poshist" in i:
                to_get.append(i)

        self._get(to_get)

    def get_spechist(self, det='g'):

        to_get = []

        for i in self._file_list:
            for j in det:
                if "spechist" in i and '_' + j + '_' in i:
                    to_get.append(i)

        self._get(to_get)
