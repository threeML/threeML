from threeML.utils.interval import Interval, IntervalSet




class TimeInterval(Interval):

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

        return self._get_mid_point()

    def __repr__(self):

        return "time interval %s - %s (duration: %s)" % (self.start_time, self.stop_time, self.duration)




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

        self.edges




