# This file contains some defaults, like locations of files, which should not
# change much but benefits anyway of being in one central location

import os


def get_user_data_path():

    user_data = os.path.join(os.path.expanduser("~"), ".threeml", "data")

    # Create it if doesn't exist
    if os.path.exists(user_data):

        return user_data

    else:

        os.makedirs(user_data)

        return user_data
