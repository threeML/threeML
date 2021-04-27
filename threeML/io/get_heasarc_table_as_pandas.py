from pathlib import Path
from typing import Union
import codecs
import datetime
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from builtins import map

import astropy.io.votable as votable
import astropy.time as astro_time
import yaml

from threeML.io.file_utils import (file_existing_and_readable,
                                   if_directory_not_existing_then_make,
                                   sanitize_filename)
from threeML.io.logging import setup_logger

log = setup_logger(__name__)


def get_heasarc_table_as_pandas(heasarc_table_name, update=False, cache_time_days=1):
    """
    Obtain a a VO table from the HEASARC archives and return it as a pandas table indexed
    by object/trigger names. The heasarc_table_name values are the ones referenced at:

    https://heasarc.gsfc.nasa.gov/docs/archive/vo/

    In order to speed up the processing of the tables, 3ML can cache the XML table in a cache
    that is updated every cache_time_days. The cache can be forced to update, i.e, reload from
    the web, by setting update to True.


    :param heasarc_table_name: the name of a HEASARC browse table
    :param update: force web read of the table and update cache
    :param cache_time_days: number of days to hold the current cache
    :return: pandas DataFrame with results and astropy table
    """

    # make sure the table is a string

    assert type(heasarc_table_name) is str

    # point to the cache directory and create it if it is not existing

    cache_directory: Path = Path("~/.threeML/.cache").expanduser()
    
    if_directory_not_existing_then_make(cache_directory)

    cache_file = cache_directory /  f"{heasarc_table_name}_cache.yml"

    cache_file_sanatized = sanitize_filename(cache_file)

    # build and sanitize the votable XML file that will be saved

    file_name = cache_directory / f"{heasarc_table_name}_votable.xml"

    file_name_sanatized = sanitize_filename(file_name)

    if not file_existing_and_readable(cache_file_sanatized):

        log.info(
            "The cache for %s does not yet exist. We will try to build it\n"
            % heasarc_table_name
        )

        write_cache = True
        cache_exists = False

    else:

        with cache_file_sanatized.open() as cache:

            # the cache file is two lines. The first is a datetime string that
            # specifies the last time the XML file was obtained

            yaml_cache = yaml.load(cache, Loader=yaml.SafeLoader)

            cached_time = astro_time.Time(
                datetime.datetime(
                    *list(map(int, yaml_cache["last save"].split("-"))))
            )

            # the second line how many seconds to keep the file around

            cache_valid_for = float(yaml_cache["cache time"])

            # now we will compare it to the current time in UTC
            current_time = astro_time.Time(
                datetime.datetime.utcnow(), scale="utc")

            delta_time = current_time - cached_time

            if delta_time.sec >= cache_valid_for:

                # ok, this is an old file, we will update it

                write_cache = True
                cache_exists = True

            else:

                # we

                write_cache = False
                cache_exists = True

    if write_cache or update:

        log.info(f"Building cache for {heasarc_table_name}")

        # go to HEASARC and get the requested table
        heasarc_url = (
            "http://heasarc.gsfc.nasa.gov/cgi-bin/W3Browse/getvotable.pl?name=%s"
            % heasarc_table_name
        )

        try:

            urllib.request.urlretrieve(
                heasarc_url, filename=file_name_sanatized)

        except (IOError):

            log.warning(
                "The cache is outdated but the internet cannot be reached. Please check your connection"
            )

        else:

            # # Make sure the lines are interpreted as Unicode (otherwise some characters will fail)
            with file_name_sanatized.open() as table_file:

                # might have to add this in for back compt J MICHAEL

                # new_lines = [x. for x in table_file.readlines()]

                new_lines = table_file.readlines()

            # now write the decoded lines back to the file
            with codecs.open(file_name_sanatized, "w+", "utf-8") as table_file:

                table_file.write("".join(new_lines))

            #        save the time that we go this table

            with open(cache_file_sanatized, "w") as cache:

                yaml_dict = {}

                current_time = astro_time.Time(
                    datetime.datetime.utcnow(), scale="utc")

                yaml_dict["last save"] = current_time.datetime.strftime(
                    "%Y-%m-%d-%H-%M-%S"
                )

                seconds_in_day = 86400.0

                yaml_dict["cache time"] = seconds_in_day * cache_time_days

                yaml.dump(yaml_dict, stream=cache, default_flow_style=False)

    # use astropy routines to read the votable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vo_table = votable.parse(str(file_name_sanatized))

    table = vo_table.get_first_table().to_table(use_names_over_ids=True)

    if table is not None:

        # make sure we do not use this as byte code
        table.convert_bytestring_to_unicode()

        # create a pandas table indexed by name

        pandas_df = table.to_pandas().set_index("name")

        del vo_table

        return pandas_df

    else:

        log.error("Your search did not return any results")

        del vo_table

        return None
