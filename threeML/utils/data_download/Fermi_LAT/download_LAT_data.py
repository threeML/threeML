from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import str
import html.parser
import re
import socket
import time
import urllib.request, urllib.parse, urllib.error
import os
import glob

import astropy.io.fits as pyfits

from threeML.io.file_utils import sanitize_filename
from threeML.config.config import threeML_config
from threeML.utils.unique_deterministic_tag import get_unique_deterministic_tag
from threeML.io.download_from_http import ApacheDirectory

# Set default timeout for operations
socket.setdefaulttimeout(120)


class DivParser(html.parser.HTMLParser):
    """
    Extract data from a <div></div> tag
    """

    def __init__(self, desiredDivName):

        html.parser.HTMLParser.__init__(self)

        self.recording = 0
        self.data = []
        self.desiredDivName = desiredDivName

    def handle_starttag(self, tag, attributes):

        if tag != "div":
            return

        if self.recording:

            self.recording += 1

            return

        for name, value in attributes:

            if name == "id" and value == self.desiredDivName:

                break

        else:

            return

        self.recording = 1

    def handle_endtag(self, tag):

        if tag == "div" and self.recording:

            self.recording -= 1

    def handle_data(self, data):

        if self.recording:

            self.data.append(data)


# Keyword name to store the unique ID for the download
_uid_fits_keyword = "QUERYUID"

def merge_LAT_data(ft1s, destination_directory=".", outfile='ft1_merged.fits'):

    outfile = os.path.join(destination_directory, outfile)

    if os.path.exists(outfile):
        print(
            "Existing merged event file %s correspond to the same selection. "
            "We assume you did not tamper with it, so we will return it instead of merging it again. "
            "If you want to redo the FT1 file again, remove it from the outdir"
            % (outfile)
        )
        return outfile

    if len(ft1s) == 1:
        print('Only one FT1 file provided. Skipping the merge...')
        import shutil
        shutil.copyfile(ft1s[0],outfile)
        return outfile

    _filelist = "_filelist.txt"

    infile = os.path.join(destination_directory, _filelist)


    infile_list = open(infile,'w')

    for ft1 in ft1s: infile_list.write(ft1 + '\n' )

    infile_list.close()

    from GtApp import GtApp

    gtselect = GtApp('gtselect')

    gtselect['infile']  = '@' + infile
    gtselect['outfile'] = outfile
    gtselect['ra']      = 'INDEF'
    gtselect['dec']     = 'INDEF'
    gtselect['rad']     = 'INDEF'
    gtselect['tmin']    = 'INDEF'
    gtselect['tmax']    = 'INDEF'
    gtselect['emin']    = '30'
    gtselect['emax']    ='1000000'
    gtselect['zmax']    = 180
    gtselect.run()
    return outfile

def download_LAT_data(
    ra,
    dec,
    radius,
    tstart,
    tstop,
    time_type,
    data_type="Photon",
    destination_directory=".",
):
    """
    Download data from the public LAT data server (of course you need a working internet connection). Data are
    selected in a circular Region of Interest (cone) centered on the provided coordinates.

    Example:

    ```
    > download_LAT_data(195.6, -35.4, 12.0, '2008-09-16 01:00:23', '2008-09-18 01:00:23',
    time_type='Gregorian', destination_directory='my_new_data')
    ```

    :param ra: R.A. (J2000) of the center of the ROI
    :param dec: Dec. (J2000) of the center of the ROI
    :param radius: radius (in degree) of the center of the ROI (use a larger radius than what you will need in the
    analysis)
    :param tstart: start time for the data
    :param tstop: stop time for the data
    :param time_type: type of the time input (one of MET, Gregorian or MJD)
    :param data_type: type of data to download. Use Photon if you use Source or cleaner classes, Extended otherwise.
    Default is Photon.
    :param destination_directory: directory where you want to save the data (default: current directory)
    :return: the path to the downloaded FT1 and FT2 file
    """
    _known_time_types = ["MET", "Gregorian", "MJD"]

    assert time_type in _known_time_types, "Time type must be one of %s" % ",".join(
        _known_time_types
    )

    valid_classes = ["Photon", "Extended"]
    assert data_type in valid_classes, "Data type must be one of %s" % ",".join(
        valid_classes
    )

    assert radius > 0, "Radius of the Region of Interest must be > 0"

    assert 0 <= ra <= 360.0, "R.A. must be 0 <= ra <= 360"
    assert -90 <= dec <= 90, "Dec. must be -90 <= dec <= 90"

    # create output directory if it does not exists
    destination_directory = sanitize_filename(destination_directory, abspath=True)

    if not os.path.exists(destination_directory):

        os.makedirs(destination_directory)

    # This will complete automatically the form available at
    # http://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi
    # After submitting the form, an html page will inform about
    # the identifier assigned to the query and the time which will be
    # needed to process it. After retrieving the query number,
    # this function will wait for the files to be completed on the server,
    # then it will download them

    url = threeML_config["LAT"]["query form"]

    # Save parameters for the query in a dictionary

    query_parameters = {}
    query_parameters["coordfield"] = "%.4f,%.4f" % (ra, dec)
    query_parameters["coordsystem"] = "J2000"
    query_parameters["shapefield"] = "%s" % radius
    query_parameters["timefield"] = "%s,%s" % (tstart, tstop)
    query_parameters["timetype"] = "%s" % time_type
    query_parameters[
        "energyfield"
    ] = "30,1000000"  # Download everything, we will chose later
    query_parameters["photonOrExtendedOrNone"] = data_type
    query_parameters["destination"] = "query"
    query_parameters["spacecraft"] = "checked"

    # Compute a unique ID for this query
    query_unique_id = get_unique_deterministic_tag(str(query_parameters))

    # Look if there are FT1 and FT2 files in the output directory matching this unique ID

    ft1s = glob.glob(os.path.join(destination_directory, "*PH??.fits"))
    ft2s = glob.glob(os.path.join(destination_directory, "*SC??.fits"))

    # Loop over all ft1s and see if there is any matching the uid

    prev_downloaded_ft1s = []
    prev_downloaded_ft2 = None

    for ft1 in ft1s:

        with pyfits.open(ft1) as f:

            this_query_uid = f[0].header.get(_uid_fits_keyword)

            if this_query_uid == query_unique_id:

                # Found one! Append to the list as there might be others

                prev_downloaded_ft1s.append(ft1)
                #break
                pass
    if len(prev_downloaded_ft1s)>0:

        for ft2 in ft2s:

            with pyfits.open(ft2) as f:

                this_query_uid = f[0].header.get(_uid_fits_keyword)

                if this_query_uid == query_unique_id:
                    # Found one! (FT2 is a single file)
                    prev_downloaded_ft2 = ft2
                    break
    else:
        # No need to look any further, if there is no FT1 file there shouldn't be any FT2 file either
        pass

    # If we have both FT1 and FT2 matching the ID, we do not need to download anymore
    if len(prev_downloaded_ft1s)>0 and prev_downloaded_ft2 is not None:

        print(
            "Existing event file %s and Spacecraft file %s correspond to the same selection. "
            "We assume you did not tamper with them, so we will return those instead of downloading them again. "
            "If you want to download them again, remove them from the outdir"
            % (prev_downloaded_ft1s, prev_downloaded_ft2)
        )

        return merge_LAT_data(prev_downloaded_ft1s, destination_directory, outfile='L%s_FT1.fits' % query_unique_id), prev_downloaded_ft2

    # Print them out

    print("Query parameters:")

    for k, v in query_parameters.items():

        print("%30s = %s" % (k, v))

    # POST encoding

    postData = urllib.parse.urlencode(query_parameters).encode("utf-8")
    temporaryFileName = "__temp_query_result.html"

    # Remove temp file if present

    try:

        os.remove(temporaryFileName)

    except:

        pass

    # This is to avoid caching

    urllib.request.urlcleanup()

    # Get the form compiled
    try:
        urllib.request.urlretrieve(url, temporaryFileName, lambda x, y, z: 0, postData)
    except socket.timeout:

        raise RuntimeError(
            "Time out when connecting to the server. Check your internet connection, or that the "
            "form at %s is accessible, then retry" % url
        )
    except Exception as e:

        print(e)

        raise RuntimeError(
            "Problems with the download. Check your internet connection, or that the "
            "form at %s is accessible, then retry" % url
        )

    # Now open the file, parse it and get the query ID

    with open(temporaryFileName) as htmlFile:

        lines = []

        for line in htmlFile:

            # lines.append(line.encode('utf-8'))
            lines.append(line)

        html = " ".join(lines).strip()

    os.remove(temporaryFileName)

    # Extract data from the response

    parser = DivParser("sec-wrapper")
    parser.feed(html)

    if parser.data == []:

        parser = DivParser("right-side")
        parser.feed(html)

    try:

        # Get line containing the time estimation

        estimatedTimeLine = [
            x
            for x in parser.data
            if x.find("The estimated time for your query to complete is") == 0
        ][0]

        # Get the time estimate

        estimatedTimeForTheQuery = re.findall(
            "The estimated time for your query to complete is ([0-9]+) seconds",
            estimatedTimeLine,
        )[0]

    except:

        raise RuntimeError(
            "Problems with the download. Empty or wrong answer from the LAT server. "
            "Please retry later."
        )

    else:

        print(
            "\nEstimated complete time for your query: %s seconds"
            % estimatedTimeForTheQuery
        )

    http_address = [
        x for x in parser.data if x.find("https://fermi.gsfc.nasa.gov") >= 0
    ][0]

    print(
        "\nIf this download fails, you can find your data at %s (when ready)\n"
        % http_address
    )

    # Now periodically check if the query is complete

    startTime = time.time()
    timeout = max(1.5 * max(5.0, float(estimatedTimeForTheQuery)), 120)  # Seconds
    refreshTime = min(float(estimatedTimeForTheQuery) / 2.0, 5.0)  # Seconds

    # precompile Url regular expression
    regexpr = re.compile("wget (.*.fits)")

    # Now download every tot seconds the status of the query, until we get status=2 (success)

    links = None
    fakeName = "__temp__query__result.html"

    while time.time() <= startTime + timeout:

        # Try and fetch the html with the results

        try:

            _ = urllib.request.urlretrieve(http_address, fakeName,)

        except socket.timeout:

            urllib.request.urlcleanup()

            raise RuntimeError(
                "Time out when connecting to the server. Check your internet connection, or that "
                "you can access %s, then retry" % threeML_config["LAT"]["query form"]
            )

        except Exception as e:

            print(e)

            urllib.request.urlcleanup()

            raise RuntimeError(
                "Problems with the download. Check your connection or that you can access "
                "%s, then retry." % threeML_config["LAT"]["query form"]
            )

        with open(fakeName) as f:

            html = " ".join(f.readlines())

        status = re.findall("The state of your query is ([0-9]+)", html)[0]

        if status == "2":

            # Success! Get the download link
            links = regexpr.findall(html)

            # Remove temp file
            os.remove(fakeName)

            # we're done
            break

        else:

            # Clean up and try again after a while

            os.remove(fakeName)

            urllib.request.urlcleanup()
            time.sleep(refreshTime)

            # Continue to next iteration

    remotePath = "%s/queries/" % threeML_config["LAT"]["public HTTP location"]

    if links != None:

        filenames = [x.split("/")[-1] for x in links]

        print("\nDownloading FT1 and FT2 files...")

        downloader = ApacheDirectory(remotePath)

        downloaded_files = [
            downloader.download(filename, destination_directory)
            for filename in filenames
        ]

    else:

        raise RuntimeError("Could not download LAT Standard data")

    # Now we need to sort so that the FT1 is always first (they might be out of order)

    # Separate the FT1 and FT2 files:

    FT1 = []
    FT2 = None

    for fits_file in downloaded_files:
        # Open the FITS file and write the unique key for this query, so that the download will not be
        # repeated if not necessary
        with pyfits.open(fits_file, mode="update") as f:

            f[0].header.set(_uid_fits_keyword, query_unique_id)

        if re.match(".+SC[0-9][0-9].fits", str(fits_file) ) is not None:

            FT2 = fits_file
        else:

            FT1.append(fits_file)

    # If FT2 is first, switch them, otherwise do nothing
    #if re.match(".+SC[0-9][0-9].fits", downloaded_files[0]) is not None:

    return merge_LAT_data(FT1, destination_directory, outfile='L%s_FT1.fits' % query_unique_id), FT2
