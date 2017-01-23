import re
from operator import itemgetter, attrgetter
import numpy as np


class IntervalsDoNotOverlap(RuntimeError):
    pass


class TimeInterval(object):

    def __init__(self, start_time, stop_time, swap_if_inverted=False):

        self._start_time = float(start_time)
        self._stop_time = float(stop_time)

        # Note that this allows to have intervals of zero duration

        if self._stop_time < self._start_time:

            if swap_if_inverted:

                self._start_time = stop_time
                self._stop_time = start_time

            else:

                raise RuntimeError("Invalid time interval! TSTART must be before TSTOP and TSTOP-TSTART >0. "
                                   "Got tstart = %s and tstop = %s" % (start_time, stop_time))

    @property
    def duration(self):

        return self._stop_time - self._start_time

    @property
    def start_time(self):

        return self._start_time

    @property
    def stop_time(self):

        return self._stop_time

    @property
    def half_time(self):

        return (self._start_time + self._stop_time) / 2.0

    def __repr__(self):

        return "time interval %s - %s (duration: %s)" % (self.start_time, self.stop_time, self.duration)

    def intersect(self, interval):
        """
        Returns a new time interval corresponding to the intersection between this interval and the provided one.

        :param interval: a TimeInterval instance
        :type interval: TimeInterval
        :return: new interval covering the intersection
        :raise IntervalsDoNotOverlap : if the intervals do not overlap
        """

        if not self.overlaps_with(interval):

            raise IntervalsDoNotOverlap("Current interval does not overlap with provided interval")

        new_start_time = max(self._start_time, interval.start_time)
        new_stop_time = min(self._stop_time, interval.stop_time)

        return TimeInterval(new_start_time, new_stop_time)

    def merge(self, interval):
        """
        Returns a new interval corresponding to the merge of the current and the provided time interval. The intervals
        must overlap.

        :param interval: a TimeInterval instance
         :type interval : TimeInterval
        :return: a new TimeInterval instance
        """

        if self.overlaps_with(interval):

            new_start_time = min(self._start_time, interval.start_time)
            new_stop_time = max(self._stop_time, interval.stop_time)

            return TimeInterval(new_start_time, new_stop_time)

        else:

            raise IntervalsDoNotOverlap("Could not merge non-overlapping intervals!")

    def overlaps_with(self, interval):
        """
        Returns whether the current time interval and the provided one overlap or not

        :param interval: a TimeInterval instance
        :type interval: TimeInterval
        :return: True or False
        """

        if interval.start_time == self._start_time or interval.stop_time == self._stop_time:

            return True

        elif interval.start_time > self._start_time and interval.start_time < self._stop_time:

            return True

        elif interval.stop_time > self._start_time and interval.stop_time < self._stop_time:

            return True

        elif interval.start_time < self._start_time and interval.stop_time > self._stop_time:

            return True

        else:

            return False

    def to_string(self):
        """
        returns a string representation of the time interval that is like the
        argument of many interval reading funcitons

        :return:
        """

        return "%f-%f"%(self.start_time,self.stop_time)

    def __add__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the right by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return TimeInterval(self._start_time + number, self._stop_time + number)

    def __sub__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the left by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return TimeInterval(self._start_time - number, self._stop_time - number)

    def __eq__(self, other):

        if not isinstance(other, TimeInterval):

            # This is needed for things like comparisons to None or other objects.
            # Of course if the other object is not even a TimeInterval, the two things
            # cannot be equal

            return False

        else:

            return self.start_time == other.start_time and self.stop_time == other.stop_time


class TimeIntervalSet(object):
    """
    A set of time intervals

    """

    def __init__(self, list_of_intervals=()):

        self._intervals = list(list_of_intervals)

    @classmethod
    def from_strings(cls, *intervals):
        """
        These are intervals specified as "-10 -- 5", "0-10", and so on

        :param intervals:
        :return:
        """


        list_of_intervals = []

        for interval in intervals:

            tmin, tmax = cls._parse_time_interval(interval)

            list_of_intervals.append(TimeInterval(tmin, tmax))

        return cls(list_of_intervals)

    @staticmethod
    def _parse_time_interval(time_interval):
        # The following regular expression matches any two numbers, positive or negative,
        # like "-10 --5","-10 - -5", "-10-5", "5-10" and so on

        tokens = re.match('(\-?\+?[0-9]+\.?[0-9]*)\s*-\s*(\-?\+?[0-9]+\.?[0-9]*)', time_interval).groups()

        return map(float, tokens)

    @classmethod
    def from_starts_and_stops(cls,start_times,stop_times):
        """
        Builds a TimeIntervalSet from a list of start and stop times:

        start = [-1,0]  ->   [-1,0], [0,1]
        stop =  [0,1]

        :param start_times:
        :param stop_times:
        :return:
        """

        assert len(start_times) == len(start_times), 'starts length: %d and stops length: %d must have same length'%(len(start_times), len(start_times))

        list_of_intervals = []

        for tmin, tmax in zip(start_times, stop_times):

            list_of_intervals.append(TimeInterval(tmin, tmax))

            return cls(list_of_intervals)

        return cls(list_of_intervals)

    @classmethod
    def from_list_of_edges(cls, time_edges):
        """
        Builds a TimeIntervalSet from a list of time edges:

        edges = [-1,0,1] -> [-1,0], [0,1]


        :param time_edges:
        :return:
        """

        assert len(time_edges)%2 != 0, "bin edges must be of unequal length"

        # sort the time edges

        time_edges.sort()

        list_of_intervals = []

        for tmin, tmax in zip(time_edges[:-1], time_edges[1:]):

            list_of_intervals.append(TimeInterval(tmin, tmax))

            return cls(list_of_intervals)

        return cls(list_of_intervals)








    def extend(self, list_of_intervals):

        self._intervals.extend(list_of_intervals)

    def __len__(self):

        return len(self._intervals)

    def __iter__(self):

        for interval in self._intervals:

            yield interval

    def __getitem__(self, item):

        return self._intervals[item]

    def __add__(self, number):
        """
        Shift all time intervals to the right by number

        :param number: a float
        :return: new TimeIntervalSet instance
        """

        new_set = TimeIntervalSet()
        new_set.extend([time_interval + number for time_interval in self._intervals])

        return new_set

    def __sub__(self, number):
        """
        Shift all time intervals to the left by number (in place)

        :param number: a float
        :return: new TimeIntervalSet instance
        """

        new_set = TimeIntervalSet([time_interval - number for time_interval in self._intervals])

        return new_set

    def pop(self, index):

        return self._intervals.pop(index)

    def sort(self):
        """
        Returns a sorted copy of the set (sorted according to the tstart of the time intervals)

        :return:
        """

        return TimeIntervalSet(itemgetter(*self.argsort())(self._intervals))

    def argsort(self):
        """
        Returns the indices which order the set

        :return:
        """

        # Gather all tstarts
        tstarts = map(lambda x:x.start_time, self._intervals)

        return map(lambda x:x[0], sorted(enumerate(tstarts), key=itemgetter(1)))

    def is_contiguous(self, relative_tolerance=1e-5):
        """
        Check whether the time intervals are all contiguous, i.e., the stop time of one interval is the start
        time of the next

        :return: True or False
        """

        start_times = map(attrgetter("start_time"), self._intervals)
        stop_times = map(attrgetter("stop_time"), self._intervals)

        return np.allclose(start_times[1:], stop_times[:-1], rtol=relative_tolerance)

    @property
    def start_times(self):

        return [start for start in self._intervals.start_time]

    def stop_times(self):

        return [stop for stop in self._intervals.stop_time]

    def to_string(self):
        """


        returns a set of string representaitons of the intervals
        :return:
        """

        return ','.join([interval.to_string() for interval in self._intervals])


