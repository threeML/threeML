# Author: G.Vianello (giacomov@stanford.edu)

import collections


class DataList(object):
    """
    A container for data sets. Can be accessed as a dictionary,
    with the [key] operator.
    """

    def __init__(self, *data_sets):
        """
        Container for data sets (i.e., plugin instances)

        :param data_sets: as many data sets as needed
        :return: (none)
        """

        self._inner_dictionary = collections.OrderedDict()

        for d in data_sets:

            if d.get_name() in self._inner_dictionary.keys():

                raise RuntimeError("You have to use unique names for data sets. %s already exists." % (d.get_name()))

            else:

                self._inner_dictionary[d.get_name()] = d

    def __setitem__(self, key, value):

        # Enforce the unique name
        if key in self.keys():

            raise RuntimeError("You have to use unique names for data sets. %s already exists." % key)

        else:

            self._inner_dictionary[key] = value

    def __getitem__(self, key):

        return self._inner_dictionary[key]

    def keys(self):

        return self._inner_dictionary.keys()

    def values(self):

        return self._inner_dictionary.values()


pass
