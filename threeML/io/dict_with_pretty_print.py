import yaml
from threeML.io.rich_display import display


class DictWithPrettyPrint(dict):
    """
    A dictionary with a _repr_html method for the Jupyter notebook

    """

    def display(self):
        return display(self)

    def _repr_html_(self):
        # yaml.dump needs a dict instance, so create one from the current content

        dumb_dict = dict(self)

        string_repr = yaml.dump(dumb_dict, default_flow_style=False)

        return '<pre>%s</pre>' % string_repr
