from builtins import range
from threeML.utils.power_of_two_utils import *


def test_is_power_of_two():
    power_of_twos = [2 ** x for x in range(32)]

    for power_of_two in power_of_twos:

        assert is_power_of_2(power_of_two)

    not_power_of_twos = [0, 3, 5, 6, 7, 9, 27, 35]

    for not_power_of_two in not_power_of_twos:

        assert is_power_of_2(not_power_of_two) == False


def test_next_power_of_two():
    assert next_power_of_2(15) == 16
    assert next_power_of_2(2) == 2
    assert next_power_of_2(65530) == 65536
