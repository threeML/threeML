import pytest

from threeML.utils.time_interval import TimeInterval, IntervalsDoNotOverlap, TimeIntervalSet


def test_time_interval_constructor():

    t = TimeInterval(-10.0, 10.0)

    assert t.start_time == -10.0
    assert t.stop_time == 10.0
    assert t.duration == 20.0
    assert t.half_time == 0.0

    with pytest.raises(RuntimeError):

        t = TimeInterval(10.0, -10.0, swap_if_inverted=False)

    t = TimeInterval(-10.0, 10.0, swap_if_inverted=True)


def test_time_interval_repr():

    t = TimeInterval(-10.0, 10.0)

    print(t)


def test_time_interval_overlaps_with():

    t1 = TimeInterval(-10.0, 10.0)
    t2 = TimeInterval(0.0, 30.0)
    t3 = TimeInterval(-100, 100.0)
    t4 = TimeInterval(-100, -10)
    t5 = TimeInterval(100.0, 200.0)

    assert t1.overlaps_with(t2) == True
    assert t1.overlaps_with(t3) == True
    assert t1.overlaps_with(t4) == False
    assert t1.overlaps_with(t5) == False


def test_time_interval_intersect():

    t1 = TimeInterval(-10.0, 10.0)
    t2 = TimeInterval(0.0, 30.0)

    t = t1.intersect(t2)

    assert t.start_time == 0.0
    assert t.stop_time == 10.0

    with pytest.raises(IntervalsDoNotOverlap):

        t1 = TimeInterval(-10.0, 10.0)
        t2 = TimeInterval(20.0, 30.0)

        _ = t1.intersect(t2)


def test_time_interval_merge():
    t1 = TimeInterval(-10.0, 10.0)
    t2 = TimeInterval(0.0, 30.0)

    t = t1.merge(t2)

    assert t.start_time == -10.0
    assert t.stop_time == 30.0

    with pytest.raises(IntervalsDoNotOverlap):

        t1 = TimeInterval(-10.0, 10.0)
        t2 = TimeInterval(20.0, 30.0)

        _ = t1.merge(t2)


def test_time_interval_add():

    t = TimeInterval(-10.0, 10.0)

    new_t = t + 10.0  # type: TimeInterval

    assert new_t.start_time == 0
    assert new_t.stop_time == 20.0


def test_time_interval_sub():

    t = TimeInterval(-10.0, 10.0)

    new_t = t - 10.0  # type: TimeInterval

    assert new_t.start_time == -20.0
    assert new_t.stop_time == 0.0


def test_time_interval_constructor_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)

    ts = TimeIntervalSet([t1, t2])

    assert ts[0] == t1
    assert ts[1] == t2

    # Use strings
    ts2 = TimeIntervalSet.from_strings("-10 - -5", "10 - 20", "20-30","-10--5")

    assert ts2[0].start_time == -10
    assert ts2[0].stop_time == -5
    assert ts2[-1].start_time == -10
    assert ts2[-1].stop_time == -5
    assert ts2[1].start_time == 10
    assert ts2[1].stop_time == 20
    assert ts2[2].start_time == 20
    assert ts2[2].stop_time == 30


def test_time_interval_iterator_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)

    ts = TimeIntervalSet([t1, t2])

    for i, tt in enumerate(ts):

        if i==0:

            assert tt == t1

        else:

            assert tt == t2


def test_time_interval_extend_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)

    ts = TimeIntervalSet([t1, t2])

    t3 = TimeInterval(30.0, 40.0)
    t4 = TimeInterval(40.0, 50.0)

    ts.extend([t3, t4])

    assert len(ts) == 4

    ts.extend(TimeIntervalSet([t3, t4]))

    assert len(ts) == 6


def test_time_interval_add_sub_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)

    ts = TimeIntervalSet([t1, t2])

    ts2 = ts + 10.0  # type: TimeIntervalSet

    assert ts2[0].start_time == 0.0
    assert ts2[1].stop_time == 40.0

    ts3 = ts - 10.0  # type: TimeIntervalSet

    assert ts3[0].start_time == -20.0
    assert ts3[1].stop_time == 20.0


def test_time_interval_argsort_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)
    t3 = TimeInterval(-30.0, 50.0)

    ts = TimeIntervalSet([t1, t2, t3])

    idx = ts.argsort()

    assert idx == [2, 0, 1]


def test_time_interval_sort_set():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)
    t3 = TimeInterval(-30.0, 50.0)

    ts = TimeIntervalSet([t1, t2, t3])

    ts2 = ts.sort()

    assert ts2[0] == t3
    assert ts2[1] == t1
    assert ts2[2] == t2


def test_time_interval_equivalence():

    t1 = TimeInterval(10.523, 20.32)

    assert t1 == TimeInterval(10.523, 20.32)

    assert not t1 == None


def test_time_interval_set_pop():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)
    t3 = TimeInterval(-30.0, 50.0)

    ts = TimeIntervalSet([t1, t2, t3])

    popped = ts.pop(1)

    assert popped == t2


def test_time_interval_set_is_contiguous():

    t1 = TimeInterval(-10.0, 20.0)
    t2 = TimeInterval(10.0, 30.0)
    t3 = TimeInterval(-30.0, 50.0)

    ts = TimeIntervalSet([t1, t2, t3])

    assert ts.is_contiguous() == False

    t1 = TimeInterval(0.0, 1.0)
    t2 = TimeInterval(1.0, 2.0)
    t3 = TimeInterval(2.0, 3.0)

    ts = TimeIntervalSet([t1, t2, t3])

    assert ts.is_contiguous() == True

    t1 = TimeInterval(0.0, 1.0)
    t2 = TimeInterval(1.1, 2.0)
    t3 = TimeInterval(2.0, 3.0)

    ts = TimeIntervalSet([t1, t2, t3])

    assert ts.is_contiguous() == False

    t1 = TimeInterval(0.0, 1.0)
    t2 = TimeInterval(2.0, 3.0)
    t3 = TimeInterval(1.0, 2.0)

    ts = TimeIntervalSet([t1, t2, t3])

    assert ts.is_contiguous() == False

    new_ts = ts.sort()

    assert new_ts.is_contiguous() == True