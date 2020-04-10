import hashlib


def get_unique_deterministic_tag(string):
    """
    Return a hex string with a one to one correspondence with the given string

    :param string: a string
    :return: a hex unique digest
    """

    return hashlib.md5(string).hexdigest()
