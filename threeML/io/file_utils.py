import os


def file_existing_and_readable(filename):
    if os.path.exists(filename):

        # Try to open it

        try:

            with open(filename):

                pass

        except:

            return False

        else:

            return True

    else:

        return False


def sanitize_filename(filename):
    return os.path.expandvars(os.path.expanduser(filename))
