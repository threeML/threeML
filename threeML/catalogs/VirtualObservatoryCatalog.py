from astromodels import *

from astropy.vo.client.vos_catalog import VOSCatalog
from astropy.vo.client import conesearch
from astropy.vo.client.exceptions import VOSError
from astropy.coordinates.name_resolve import get_icrs_coordinates


class VirtualObservatoryCatalog(object):
    
    def __init__(self, name, url, description):
                
        self.catalog = VOSCatalog.create(name, url, description=description)

        self._last_query_results = None

    def search_around_source(self, source_name, radius):
        """
        Search for sources around the named source. The coordinates of the provided source are resolved using the
        astropy.coordinates.name_resolve facility.

        :param source_name: name of the source, like "Crab"
        :param radius: radius of the search, in degrees
        :return: (ra, dec, table), where ra,dec are the coordinates of the source as resolved by astropy, and table is
        a table with the list of sources
        """

        sky_coord = get_icrs_coordinates(source_name)

        ra, dec = (sky_coord.fk5.ra.value, sky_coord.fk5.dec.value)

        return ra, dec, self.cone_search(ra, dec, radius)

    def cone_search(self, ra, dec, radius):
        """
        Searches for sources in a cone of given radius and center

        :param ra: decimal degrees, R.A. of the center of the cone
        :param dec: decimal degrees, Dec. of the center of the cone
        :param radius: radius in degrees
        :return: a table with the list of sources
        """

        skycoord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')

        with warnings.catch_warnings():
            
            #Ignore all warnings, which are many from the conesearch module
            
            warnings.simplefilter('ignore')
            
            try:
                
                votable = conesearch.conesearch(skycoord, radius, 
                                                catalog_db=self.catalog,
                                                verb=3, verbose=True,
                                                cache=False)
            
            except VOSError as exc:
                
                print(exc.message)
                return None

        table = votable.to_table()

        self._last_query_results = table.to_pandas().set_index('name').sort_values("Search_Offset")

        out = self.apply_format(table)
        
        #This is needed to avoid strange errors
        del votable
        del table

        # Save coordinates of center of cone search
        self._ra = ra
        self._dec = dec

        # Make a DataFrame with the name of the source as index

        return out

    @property
    def ra_center(self):
        return self._ra

    @property
    def dec_center(self):
        return self._dec

    def apply_format(self, table):

        raise NotImplementedError("You have to override this!")

    def get_model(self):

        raise NotImplementedError("You have to override this!")
