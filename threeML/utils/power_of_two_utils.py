def is_power_of_2(num):
    """
    Returns whether num is a power of two or not
    :param num: an integer positive number
    :return: True if num is a power of 2, False otherwise
    """

    return num != 0 and ((num & (num - 1)) == 0)


def next_power_of_2(x):
    """
    Returns the first power of two >= x, so f(2) = 2, f(127) = 128, f(65530) = 65536

    :param x:
    :return:
    """

    # NOTES for this black magic:
    # * .bit_length returns the number of bits necessary to represent self in binary
    # * x << y means 1 with the bits shifted to the left by y, which is the same as multiplying x by 2**y (but faster)

    return 1 << (x - 1).bit_length()
