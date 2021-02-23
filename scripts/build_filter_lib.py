import xml.etree.ElementTree as ET
import urllib
from collections import defaultdict
from pathlib import Path
import warnings
import h5py
import time
import astropy.io.votable as votable
import astropy.units as u
import numpy as np
import pandas as pd
import speclite.filters as spec_filter
import io
import re
from threeML.utils.photometry.filter_library import get_speclite_filter_library
from threeML.io.network import internet_connection_is_active
from threeML.io.file_utils import file_existing_and_readable





def to_valid_python_name(name):

    new_name = name.replace("-", "_")

    try:

        int(new_name[0])

        new_name = "f_%s" % new_name

        return new_name

    except (ValueError):

        return new_name


def add_svo_filter_to_speclite(observatory, instrument, ffilter, update=False):
    """
    download an SVO filter file and then add it to the user library
    :param observatory:
    :param instrument:
    :param ffilter:
    :return:
    """


    if True:

        url_response = urllib.request.urlopen(
            "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?PhotCalID=%s/%s.%s/AB"
            % (observatory.replace(" ", "%20"), instrument, ffilter)
        )
        # now parse it
        data = votable.parse_single_table(io.BytesIO(url_response.read())).to_table()



        
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
                wavelength=data["Wavelength"] *
                waveunit.to("Angstrom") * u.Angstrom,
                response=data["Transmission"],
                meta=dict(
                    group_name=to_valid_python_name(instrument),
                    band_name=to_valid_python_name(ffilter),
                ),
            )

            observatory = observatory.replace(" ", "")
            
            with h5py.File(get_speclite_filter_library(), 'a') as f:

                if observatory not in f.keys():

                    obs_grp = f.create_group(observatory)

                else:

                    obs_grp = f[observatory]

                
                grp_name = to_valid_python_name(instrument)
                
                if grp_name not in obs_grp.keys():

                    grp = obs_grp.create_group(grp_name)
                
                else:

                    grp = obs_grp[grp_name]

                band_name = to_valid_python_name(ffilter)

                if band_name not in grp.keys():

                    sub_grp = grp.create_group(band_name)

                else:

                    sub_grp = grp[band_name]

                sub_grp.create_dataset("wavelength",
                                         data=(data["Wavelength"]*waveunit.to("Angstrom")), compression="gzip")
            
                sub_grp.create_dataset("transmission",data=data["Transmission"],compression="gzip")

                
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

    
    with h5py.File(get_speclite_filter_library(), "a") as f:

        f.attrs["start"] = 1
        

    
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

    for obs in observatories[::-1][50:]:

        
        time.sleep(1)
        # fix 2MASS to a valid name

        # if obs == "La Silla":
        # #     continue
        #     obs = 
        url = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?Facility=%s" % obs.replace(" ", "%20")

        url_response = urllib.request.urlopen(url)

        try:

            # parse the VO table

            v = votable.parse(io.BytesIO(url_response.read()))

            instrument_dict = defaultdict(list)

            # get the filter names for this observatory

            instruments = v.get_first_table().to_table()["filterID"].tolist()

            print("Downloading %s filters" % (obs))

            for x in instruments:

                

                
                
                _, instrument, subfilter = search_name.match(x).groups()

                go = True

                with h5py.File(get_speclite_filter_library(),"r") as f:
                    
                    if obs in f.keys():

                        if to_valid_python_name(instrument) in f[obs].keys():

                            if to_valid_python_name(subfilter) in f[obs][to_valid_python_name(instrument)].keys():

                                go =False


                        
                if go:

                    print(f"now on {obs} {instrument} {subfilter}")
                    success = add_svo_filter_to_speclite(
                        obs, instrument, subfilter, update)

                else:

                    success = True
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


    grond_filter_url = "http://www.mpe.mpg.de/~jcg/GROND/GROND_filtercurves.txt"

    url_response = urllib.request.urlopen(grond_filter_url)

    grond_table = pd.read_table(url_response)

    wave = np.array(grond_table["A"])

    bands = ["g", "r", "i", "z", "H", "J", "K"]

    with h5py.File(get_speclite_filter_library(),"r+") as f:

        this_grp = f["LaSilla"]

        try:
            this_ins = this_grp.create_group("GROND")

        except:

            this_ins = this_grp["GROND"]
    

        for band in bands:

            curve = np.array(grond_table["%sBand" % band])
            curve[curve < 0] = 0
            curve[0] = 0
            curve[-1] = 0

            wavelength = wave * u.nm

            wavelength = wavelength.to(u.angstrom).value

            band_grp = this_ins.create_group(band)
            
            band_grp.create_dataset("wavelength", data=wavelength, compression="gzip")

            band_grp.create_dataset("transmission", data=curve, compression="gzip")
            

    filter_dict["ESO"] = {"GROND": bands}

    return filter_dict


update = False

def build_filter_library():

    if not file_existing_and_readable(get_speclite_filter_library()) or update:

        print("Downloading optical filters. This will take a while.\n")

        if internet_connection_is_active():

            filter_dict = {}


            
            filter_dict = download_grond(filter_dict)

            
            filter_dict = download_SVO_filters(filter_dict)

            # filter_dict = download_grond(filter_dict)

            # # ok, finally, we want to keep track of the svo filters we have
            # # so we will save this to a yaml file for future reference
            # with open(
            #     os.path.join(get_speclite_filter_path(), "filter_lib.yml"), "w"
            # ) as f:

            #     yaml.safe_dump(filter_dict, f, default_flow_style=false)

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
