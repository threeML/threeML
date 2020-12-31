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

            if d.name in self._inner_dictionary.keys():

                raise RuntimeError(
                    "You have to use unique names for data sets. %s already exists."
                    % (d.name)
                )

            else:

                self._inner_dictionary[d.name] = d

    def insert(self, dataset):

        # Enforce the unique name
        if dataset.name in self.keys():

            raise RuntimeError(
                "You have to use unique names for data sets. %s already exists." % dataset.name
            )

        else:

            self._inner_dictionary[dataset.name] = dataset

    def __getitem__(self, key):

        return self._inner_dictionary[key]

    def keys(self):

        return self._inner_dictionary.keys()

    def values(self):

        return self._inner_dictionary.values()
