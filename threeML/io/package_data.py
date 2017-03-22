import pkg_resources


def get_path_of_data_file(data_file):
    file_path = pkg_resources.resource_filename("threeML", 'data/%s' % data_file)

    return file_path



def get_path_of_data_dir():
    file_path = pkg_resources.resource_filename("threeML", 'data')

    return file_path