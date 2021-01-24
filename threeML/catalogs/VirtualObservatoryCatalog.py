
from astromodels import *
import astropy

# from astropy.vo.client.vos_catalog import VOSCatalog
from astroquery.vo_conesearch.vos_catalog import VOSCatalog
from astroquery.vo_conesearch import conesearch
from astroquery.vo_conesearch.exceptions import VOSError


from astropy.coordinates.name_resolve import get_icrs_coordinates

import astropy.table as astro_table

from threeML.io.network import internet_connection_is_active
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

# Workaround to support astropy 4.1+
astropy_old = True
astropy_version = astropy.__version__
if int(astropy_version[0]) >= 4 and int(astropy_version[2]) >= 1:
    astropy_old = False

class ConeSearchFailed(RuntimeError):
    pass


class VirtualObservatoryCatalog(object):
    def __init__(self, name, url, description):

        self.catalog = VOSCatalog.create(name, url, description=description)

        self._get_vo_table_from_source()

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

        skycoord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")

        # First check that we have an active internet connection
        if not internet_connection_is_active():  # pragma: no cover

            raise ConeSearchFailed(
                "It looks like you don't have an active internet connection. Cannot continue."
            )

        with warnings.catch_warnings():

            # Ignore all warnings, which are many from the conesearch module

            warnings.simplefilter("ignore")

            try:

                votable = conesearch.conesearch(
                    skycoord,
                    radius,
                    catalog_db=self.catalog,
                    verb=3,
                    verbose=True,
                    cache=False,
                )

            except VOSError as exc:  # Pragma: no cover

                # Download failed

                raise ConeSearchFailed("Cone search failed. Reason: %s" % exc.message)

            else:

                # Download successful
                table = votable
                # Workaround to comply with newer versions of astroquery
                if isinstance(votable, astropy.io.votable.tree.Table):
                    table = votable.to_table()

                if table is None:

                    log.error("Your search returned nothing")
                    
                    return None
                    
                table.convert_bytestring_to_unicode()

                pandas_df = (
                    table.to_pandas().set_index("name").sort_values("Search_Offset")
                )

                str_df = pandas_df.select_dtypes([np.object])
                
                if astropy_old:
                    str_df = str_df.stack().str.decode("utf-8").unstack()
                
                for col in str_df:
                    pandas_df[col] = str_df[col]

                if astropy_old:
                    new_index = [x.decode("utf-8") for x in pandas_df.index]
                    pandas_df.index = new_index

                self._last_query_results = pandas_df
                
                out = self.apply_format(table)

                # This is needed to avoid strange errors
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

    def _get_vo_table_from_source(self):

        raise NotImplementedError("You have to override this!")

    def query(self, query):
        """
        query the entire VO table for the given logical argument. Queries are in the form of pandas
        queries: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

        To obtain a preview of the availble columns, try catalog.variables


        :param query: pandas style query string
        :return:
        """

        assert type(query) == str, "query must be a string"

        query_results = self._vo_dataframe.query(query)

        table = astro_table.Table.from_pandas(query_results)
        name_column = astro_table.Column(name="name", data=query_results.index)
        table.add_column(name_column, index=0)

        out = self.apply_format(table)

        self._last_query_results = query_results

        return out

    def query_sources(self, *sources):
        """
        query for the specific source names.

        :param sources: source(s) to search for
        :return:
        """

        valid_sources = []

        for source in sources:

            if self._source_is_valid(source):

                valid_sources.append(source)

        if valid_sources:

            query_string = " | ".join(['(index == "%s")' % x for x in valid_sources])

            query_results = self._vo_dataframe.query(query_string)

            table = astro_table.Table.from_pandas(query_results)

            name_column = astro_table.Column(name="name", data=query_results.index)
            table.add_column(name_column, index=0)

            out = self.apply_format(table)

            self._last_query_results = query_results

            return out

        else:

            RuntimeError("There were not valid sources in your search")

    def _source_is_valid(self, source):

        raise NotImplementedError("You have to override this!")

    @property
    def result(self):
        """
        return a searchable pandas dataframe of results from the last query.
        :return:
        """

        return self._last_query_results.copy(deep=True)
