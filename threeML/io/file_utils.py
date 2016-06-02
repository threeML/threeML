import os


def file_existing_and_readable(filename):

    sanitized_filename = sanitize_filename(filename)

    if os.path.exists(sanitized_filename):

        # Try to open it

        try:

            with open(sanitized_filename):

                pass

        except:

            return False

        else:

            return True

    else:

        return False


def sanitize_filename(filename):
    return os.path.expandvars(os.path.expanduser(filename))
