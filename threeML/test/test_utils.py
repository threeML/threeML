from threeML.utils.fermi_relative_mission_time import (
    compute_fermi_relative_mission_times,
)


def test_fermi_relative_mission_times():
    res = compute_fermi_relative_mission_times(0)
    assert res is None

    res = compute_fermi_relative_mission_times("524666471")
    assert int(res["Fermi seconds since 2001.0 UTC (decimal)"]) == 524666471
    assert res["LIGO/GPS seconds since 1980-01-06 UTC (decimal)"] == "1187008884.000"
