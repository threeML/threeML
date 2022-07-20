from builtins import object
import pandas as pd
import numpy as np
from threeML.io.table import long_path_formatter
from threeML.io.rich_display import display
from threeML.io.uncertainty_formatter import uncertainty_formatter


class ResultsTable(object):
    def __init__(
        self, parameter_paths, values, negative_errors, positive_errors, units
    ):

        values_s = pd.Series([], dtype=np.float64)
        negative_error_s = pd.Series([], dtype=np.float64)
        positive_error_s = pd.Series([], dtype=np.float64)
        units_s = pd.Series([], dtype=np.float64)

        for i, this_path in enumerate(parameter_paths):

            # Check if this parameter has a dex() unit, i.e., if it is in log10 scale
            # If it is, we display the transformed value, not the logarithm

            units_s[this_path] = units[i]

            if units_s[this_path].to_string().find("dex") < 0:

                # A normal parameter
                values_s[this_path] = values[i]
                negative_error_s[this_path] = negative_errors[i]
                positive_error_s[this_path] = positive_errors[i]

            else:

                # A dex() parameter (logarithmic parameter)
                values_s[this_path] = 10 ** values[i]
                negative_error_s[this_path] = (
                    10 ** (values[i] + negative_errors[i]) - values_s[this_path]
                )
                positive_error_s[this_path] = (
                    10 ** (values[i] + positive_errors[i]) - values_s[this_path]
                )

        self._data_frame = pd.DataFrame()
        self._data_frame["value"] = values_s
        self._data_frame["negative_error"] = negative_error_s
        self._data_frame["positive_error"] = positive_error_s
        self._data_frame["error"] = (
            np.abs(negative_error_s.values) + positive_error_s.values
        ) / 2.0
        self._data_frame["unit"] = units_s

    @property
    def frame(self):

        return self._data_frame

    def display(self, key_formatter=long_path_formatter):
        def row_formatter(row):

            value = row["value"]
            lower_bound = value + row["negative_error"]
            upper_bound = value + row["positive_error"]

            pretty_string = uncertainty_formatter(value, lower_bound, upper_bound)

            return pretty_string

        # Make another data frame with the keys
        new_frame = self._data_frame.copy(deep=True)  # type: pd.DataFrame

        # Add new column which will become the new index
        new_frame["parameter"] = [key_formatter(x) for x in new_frame.index.values]

        # Set it as the index
        new_frame.set_index("parameter", drop=True, inplace=True)

        # compute the display
        new_frame["result"] = new_frame.apply(row_formatter, axis=1)

        # Display

        display(new_frame[["result", "unit"]])
