def dash_separated_string_to_tuple(arg):
    """
    turn a dash separated string into a tuple
    
    :param arg: a dash separated string "a-b"
    :return: (a,b)
    """

    return arg.replace(" ", "").split("-")
