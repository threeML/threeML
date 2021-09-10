from __future__ import print_function

import glob
import html.parser
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from builtins import str
from pathlib import Path

import astropy.io.fits as pyfits

from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import TimeTypeNotKnown
from threeML.io.download_from_http import ApacheDirectory
from threeML.io.file_utils import sanitize_filename
from threeML.io.logging import setup_logger
from threeML.utils.unique_deterministic_tag import get_unique_deterministic_tag

log = setup_logger(__name__)

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


def merge_LAT_data(ft1s, destination_directory: str = ".", outfile: str = 'ft1_merged.fits', Emin: float = 30.0, Emax: float = 1e6) -> Path:

    outfile: Path = Path(destination_directory) / outfile

    if outfile.exists():
        log.warning(
            f"Existing merged event file {outfile} correspond to the same selection. "
            "We assume you did not tamper with it, so we will return it instead of merging it again. "
            "If you want to redo the FT1 file again, remove it from the outdir"

        )
        return outfile

    if len(ft1s) == 1:

        log.warning('Only one FT1 file provided. Skipping the merge...')
        import shutil

        os.rename(ft1s[0], outfile)
        return outfile

    _filelist = "_filelist.txt"

    infile: Path = Path(destination_directory) / _filelist

    infile_list = infile.open('w')

    for ft1 in ft1s:
        infile_list.write(str(ft1) + '\n')

    infile_list.close()

    from GtApp import GtApp

    gtselect = GtApp('gtselect')

    gtselect['infile'] = '@' + str(infile)
    gtselect['outfile'] = str(outfile)
    gtselect['ra'] = 'INDEF'
    gtselect['dec'] = 'INDEF'
    gtselect['rad'] = 'INDEF'
    gtselect['tmin'] = 'INDEF'
    gtselect['tmax'] = 'INDEF'
    gtselect['emin'] = '%.3f' % Emin
    gtselect['emax'] = '%.3f' % Emax
    gtselect['zmax'] = 180
    gtselect.run()
    return outfile


def download_LAT_data(
    ra: float,
    dec: float,
    radius: float,
    tstart: float,
    tstop: float,
    time_type: str,
    data_type: str = "Photon",
    destination_directory: str = ".",
    Emin: float = 30.,
    Emax: float = 1000000.
) -> Path:
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
    :param Emin: minimum photon energy (in MeV) to download (default: 30 MeV, must be between 30 and 1e6 MeV)
    :param Emax: maximum photon energy (in MeV) to download (default: 1e6 MeV, must be betwen 30 and 1e6 MeV )
    :return: the path to the downloaded FT1 and FT2 file
    """
    _known_time_types = ["MET", "Gregorian", "MJD"]

    if time_type not in _known_time_types:
        out = ",".join(_known_time_types)
        log.error(
            f"Time type must be one of {out}"
        )
        raise TimeTypeNotKnown()

    valid_classes = ["Photon", "Extended"]
    if data_type not in valid_classes:
        out = ",".join(valid_classes)
        log.error(
            f"Data type must be one of {out}"
        )
        raise TypeError()

    if radius <= 0:
        log.error(
            "Radius of the Region of Interest must be > 0"
        )
        raise ValueError()

    if not (0 <= ra <= 360.0):
        log.error(
            "R.A. must be 0 <= ra <= 360"
        )
        raise ValueError()

    if not -90 <= dec <= 90:
        log.error(
            "Dec. must be -90 <= dec <= 90"
        )
        raise ValueError()

    fermiEmin = 30
    fermiEmax = 1e6
    
    if Emin < fermiEmin:
        log.warning( f"Setting Emin from {Emin} to 30 MeV (minimum available energy for Fermi-LAT data)" )
        Emin = fermiEmin
        
    if Emin > fermiEmax:
        log.warning( f"Setting Emin from {Emin} to 1 TeV (maximum available energy for Fermi-LAT data)" )
        Emin = fermiEmax
    
    if Emax < fermiEmin:
        log.warning( f"Setting Emax from {Emax} to 30 MeV (minimum available energy for Fermi-LAT data)" )
        Emax = fermiEmin
        
    if Emax > fermiEmax:
        log.warning( f"Setting Emax from {Emax} to 1 TeV (maximum available energy for Fermi-LAT data)" )
        Emax = fermiEmax

    if Emin >= Emax:
        log.error( f"Minimum energy ({Emin}) must be less than maximum energy ({Emax}) for download." )
        raise ValueError()
        

    # create output directory if it does not exists
    destination_directory = sanitize_filename(
        destination_directory, abspath=True)

    if not destination_directory.exists():

        destination_directory.mkdir(parents=True)

    # This will complete automatically the form available at
    # http://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi
    # After submitting the form, an html page will inform about
    # the identifier assigned to the query and the time which will be
    # needed to process it. After retrieving the query number,
    # this function will wait for the files to be completed on the server,
    # then it will download them

    url: str = threeML_config.LAT.query_form

    # Save parameters for the query in a dictionary

    query_parameters = {}
    query_parameters["coordfield"] = "%.4f,%.4f" % (ra, dec)
    query_parameters["coordsystem"] = "J2000"
    query_parameters["shapefield"] = "%s" % radius
    query_parameters["timefield"] = "%s,%s" % (tstart, tstop)
    query_parameters["timetype"] = "%s" % time_type
    query_parameters["energyfield"] = "%.3f,%.3f" % (Emin, Emax)
    query_parameters["photonOrExtendedOrNone"] = data_type
    query_parameters["destination"] = "query"
    query_parameters["spacecraft"] = "checked"

    # Print them out

    log.info("Query parameters:")

    for k, v in query_parameters.items():

        log.info("%30s = %s" % (k, v))


    # Compute a unique ID for this query
    query_unique_id = get_unique_deterministic_tag(str(query_parameters))
    log.info( "Query ID: %s" % query_unique_id)

    # Look if there are FT1 and FT2 files in the output directory matching this unique ID

    ft1s = [x for x in destination_directory.glob("*PH??.fits")]
    ft2s = [x for x in destination_directory.glob("*SC??.fits")]

    # Loop over all ft1s and see if there is any matching the uid

    prev_downloaded_ft1s = []
    prev_downloaded_ft2 = None

    for ft1 in ft1s:

        with pyfits.open(ft1) as f:

            this_query_uid = f[0].header.get(_uid_fits_keyword)

            if this_query_uid == query_unique_id:

                # Found one! Append to the list as there might be others

                prev_downloaded_ft1s.append(ft1)
                # break
                pass
    if len(prev_downloaded_ft1s) > 0:

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
    if len(prev_downloaded_ft1s) > 0 and prev_downloaded_ft2 is not None:

        log.warning(
            f"Existing event file {prev_downloaded_ft1s} and Spacecraft file {prev_downloaded_ft2} correspond to the same selection. "
            "We assume you did not tamper with them, so we will return those instead of downloading them again. "
            "If you want to download them again, remove them from the outdir"

        )

        return (
            merge_LAT_data(
                prev_downloaded_ft1s,
                destination_directory,
                outfile="L%s_FT1.fits" % query_unique_id,
                Emin = Emin,
                Emax = Emax
            ),
            prev_downloaded_ft2,
        )


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
        urllib.request.urlretrieve(
            url, temporaryFileName, lambda x, y, z: 0, postData)
    except socket.timeout:

        log.error(
            "Time out when connecting to the server. Check your internet connection, or that the "
            f"form at {url} is accessible, then retry"
        )
        raise RuntimeError(

        )
    except Exception as e:

        log.error(e)
        log.exception("Problems with the download. Check your internet connection, or that the "
                      f"form at {url} is accessible, then retry")

        raise RuntimeError(

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

        estimated_time_for_the_query = re.findall(
            "The estimated time for your query to complete is ([0-9]+) seconds",
            estimatedTimeLine,
        )[0]

    except:

        raise RuntimeError(
            "Problems with the download. Empty or wrong answer from the LAT server. "
            "Please retry later."
        )

    else:

        log.info(
            f"Estimated complete time for your query: {estimated_time_for_the_query} seconds"

        )

    http_address = [
        x for x in parser.data if x.find("https://fermi.gsfc.nasa.gov") >= 0
    ][0]

    log.info(
        f"If this download fails, you can find your data at {http_address} (when ready)"

    )

    # Now periodically check if the query is complete

    startTime = time.time()
    timeout = max(
        1.5 * max(5.0, float(estimated_time_for_the_query)), 120)  # Seconds
    refreshTime = min(float(estimated_time_for_the_query) /
                      2.0, 5.0)  # Seconds

    # precompile Url regular expression
    regexpr = re.compile("wget (.*.fits)")

    # Now download every tot seconds the status of the query, until we get status=2 (success)

    links = None
    fakeName = "__temp__query__result.html"

    while time.time() <= startTime + timeout:

        # Try and fetch the html with the results

        try:

            _ = urllib.request.urlretrieve(
                http_address,
                fakeName,
            )

        except socket.timeout:

            urllib.request.urlcleanup()

            log.exception(
                "Time out when connecting to the server. Check your internet connection, or that "
                f"you can access {threeML_config.LAT.query_form}, then retry")

            raise RuntimeError(
            )

        except Exception as e:

            log.error(e)

            urllib.request.urlcleanup()

            log.exception("Problems with the download. Check your connection or that you can access "
                          f"{threeML_config.LAT.query_form}, then retry.")

            raise RuntimeError(

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

    remotePath = "%s/queries/" % threeML_config.LAT.public_http_location

    if links != None:

        filenames = [x.split("/")[-1] for x in links]

        log.info("Downloading FT1 and FT2 files...")

        downloader = ApacheDirectory(remotePath)

        downloaded_files = [
            downloader.download(filename, destination_directory)
            for filename in filenames
        ]

    else:

        log.error(
            "Could not download LAT Standard data"
        )

        raise RuntimeError()

    # Now we need to sort so that the FT1 is always first (they might be out of order)

    # Separate the FT1 and FT2 files:

    FT1 = []
    FT2 = None

    for fits_file in downloaded_files:
        # Open the FITS file and write the unique key for this query, so that the download will not be
        # repeated if not necessary
        with pyfits.open(fits_file, mode="update") as f:

            f[0].header.set(_uid_fits_keyword, query_unique_id)

        if re.match(".+SC[0-9][0-9].fits", str(fits_file)) is not None:

            FT2 = fits_file
        else:

            FT1.append(fits_file)

    # If FT2 is first, switch them, otherwise do nothing
    # if re.match(".+SC[0-9][0-9].fits", downloaded_files[0]) is not None:

    return (
        merge_LAT_data(
            FT1,
            destination_directory,
            outfile="L%s_FT1.fits" % query_unique_id,
            Emin = Emin,
            Emax = Emax
        ),
        FT2
    )

class LAT_dataset():

    def __init__(self):
        self.ft1=None
        self.ft2=None
        pass

    def make_LAT_dataset(self,
                         ra: float,
                         dec: float,
                         radius: float,
                         trigger_time : float,
                         tstart: float,
                         tstop: float,
                         data_type: str = "Photon",
                         destination_directory: str = ".",
                         Emin: float = 30.,
                         Emax: float = 1000000.):

        self.trigger_time = trigger_time
        self.ra           = ra
        self.dec          = dec
        self.METstart     = tstart+trigger_time
        self.METstop      = tstop+trigger_time
        self.Emin         = Emin
        self.Emax         = Emax

        self.destination_directory = destination_directory


        import datetime
        from GtBurst.dataHandling import met2date,_makeDatasetsOutOfLATdata

        metdate = 239241601

        if tstart>metdate: assert("Start time must bge relative to triggertime")
        if tstop>metdate:  assert("Stop time must bge relative to triggertime")

        grb_name = met2date(trigger_time, opt='grbname')

        destination_directory = os.path.join(destination_directory,'bn%s' % grb_name)

        new_ft1 = os.path.join(destination_directory, "gll_%s_tr_bn%s_v00.fit" % ('ft1', grb_name))

        new_ft2 = os.path.join(destination_directory, "gll_%s_tr_bn%s_v00.fit" % ('ft2', grb_name))

        eboundsFilename = os.path.join(destination_directory, "gll_%s_tr_bn%s_v00.rsp" % ('cspec', grb_name))

        if (not os.path.exists(new_ft1) or not os.path.exists(new_ft2) or not os.path.exists(eboundsFilename)) :
            ft1,ft2 = download_LAT_data(
                                        ra,
                                        dec,
                                        radius,
                                        trigger_time + tstart,
                                        trigger_time + tstop,
                                        time_type='MET',
                                        data_type=data_type,
                                        destination_directory=destination_directory,
                                        Emin=Emin,
                                        Emax=Emax
            )


            os.rename(str(ft1), new_ft1 )

            os.rename(str(ft2), new_ft2 )

            _, eboundsFilename, _, cspecfile = _makeDatasetsOutOfLATdata(new_ft1, new_ft2,
                                                                             grb_name,
                                                                             tstart, tstop,
                                                                             ra, dec,
                                                                             trigger_time,
                                                                             destination_directory,
                                                                             cspecstart=tstart,
                                                                             cspecstop=tstop)
        self.grb_name = grb_name
        self.ft1      = new_ft1
        self.ft2      = new_ft2
        self.rspfile     = eboundsFilename
        pass


    def extract_events(self,roi, zmax, irf, thetamax=180.0,strategy='time'):
        from GtBurst import dataHandling
        global lastDisplay

        LATdata = dataHandling.LATData(self.ft1, self.rspfile, self.ft2)

        self.filt_file, nEvents = LATdata.performStandardCut(self.ra, self.dec, roi, irf, self.METstart, self.METstop, self.Emin, self.Emax, zmax,
                                                           thetamax,
                                                           True, strategy=strategy.lower())
        log.info('Extracted %s events' % nEvents)


