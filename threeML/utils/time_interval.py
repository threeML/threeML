from threeML.utils.interval import Interval, IntervalSet
from threeML.io.rich_display import display

import collections
import pandas as pd


class TimeInterval(Interval):
    def __add__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the right by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return self.new(self._start + number, self._stop + number)

    def __sub__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the left by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return self.new(self._start - number, self._stop - number)

    @property
    def duration(self):

        return super(TimeInterval, self)._get_width()

    @property
    def start_time(self):

        return self._start

    @property
    def stop_time(self):

        return self._stop

    @property
    def half_time(self):

        return self.mid_point

    def __repr__(self):

        return "time interval %s - %s (duration: %s)" % (
            self.start_time,
            self.stop_time,
            self.duration,
        )


class TimeIntervalSet(IntervalSet):
    """
    A set of time intervals

    """

    INTERVAL_TYPE = TimeInterval

    @property
    def start_times(self):
        """
        Return the starts fo the set

        :return: list of start times
        """

        return self.starts

    @property
    def stop_times(self):
        """
        Return the stops of the set

        :return:
        """

        return self.stops

    @property
    def absolute_start_time(self):
        """
        the minimum of the start times
        :return:
        """

        return self.absolute_start

    @property
    def absolute_stop_time(self):
        """
        the maximum of the stop times
        :return:
        """

        return self.absolute_stop

    @property
    def time_edges(self):
        """
        return an array of time edges if contiguous
        :return:
        """

        return self.edges

    def __add__(self, number):
        """
        Shift all time intervals to the right by number

        :param number: a float
        :return: new TimeIntervalSet instance
        """

        new_set = self.new()
        new_set.extend([time_interval + number for time_interval in self._intervals])

        return new_set

    def __sub__(self, number):
        """
        Shift all time intervals to the left by number (in place)

        :param number: a float
        :return: new TimeIntervalSet instance
        """

        new_set = self.new(
            [time_interval - number for time_interval in self._intervals]
        )

        return new_set

    def _create_pandas(self):

        time_interval_dict = collections.OrderedDict()

        time_interval_dict["Start"] = []
        time_interval_dict["Stop"] = []
        time_interval_dict["Duration"] = []
        time_interval_dict["Midpoint"] = []

        for i, interval in enumerate(self._intervals):

            time_interval_dict["Start"].append(interval.start)
            time_interval_dict["Stop"].append(interval.stop)
            time_interval_dict["Duration"].append(interval.duration)
            time_interval_dict["Midpoint"].append(interval.half_time)

        df = pd.DataFrame(data=time_interval_dict)

        return df

    def display(self):
        """
        Display the time intervals

        :return: None
        """

        display(self._create_pandas())

    def __repr__(self):

        return self._create_pandas().to_string()
