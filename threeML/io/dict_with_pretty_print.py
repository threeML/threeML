from __future__ import print_function
import yaml
from threeML.io.rich_display import display
import collections


class DictWithPrettyPrint(collections.OrderedDict):
    """
    A dictionary with a _repr_html method for the Jupyter notebook

    """

    def display(self):
        return display(self)

    def __str__(self):

        string_repr = yaml.dump(dict(self), default_flow_style=False)

        return string_repr

    def _repr_pretty_(self, pp, cycle):

        print(self.__str__())

    def _repr_html_(self):

        string_repr = self.__str__()

        return "<pre>%s</pre>" % string_repr
