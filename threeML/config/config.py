import os
import shutil
import re
import pkg_resources
import yaml
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from threeML.exceptions.custom_exceptions import custom_warnings, ConfigurationFileCorrupt


_config_file_name = 'threeML_config.yml'


def get_path_of_default_configuration():

    file_path = pkg_resources.resource_filename("threeML", 'config/%s' % _config_file_name)

    return file_path


class Config(object):

    def __init__(self):

        # Read first the default configuration file
        default_configuration_path = get_path_of_default_configuration()

        assert os.path.exists(default_configuration_path), \
            "Default configuration %s does not exist. Re-install 3ML" % default_configuration_path

        with open(default_configuration_path) as f:

            try:

                configuration = yaml.safe_load(f)

            except:

                raise ConfigurationFileCorrupt("Default configuration file %s cannot be parsed!" %
                                               (default_configuration_path))

            # This needs to be here for the _check_configuration to work

            self._default_configuration = None

            # Test the default configuration

            try:

                self._default_configuration = self._check_configuration(configuration, default_configuration_path)

            except:

                raise

            else:

                self._default_path = default_configuration_path

        # Check if the user has a user-supplied config file under .threeML

        user_config_path = os.path.join(os.path.expanduser('~'), '.threeML', _config_file_name)

        if os.path.exists(user_config_path):

            with open(user_config_path) as f:

                configuration = yaml.safe_load(f)

                # Test if the local/configuration is ok

                try:

                    self._configuration = self._check_configuration(configuration, user_config_path)

                except ConfigurationFileCorrupt:

                    # Probably an old configuration file
                    custom_warnings.warn("The user configuration file at %s does not appear to be valid. We will "
                                         "substitute it with the default configuration. You will find a copy of the "
                                         "old configuration at %s so you can transfer any customization you might "
                                         "have from there to the new configuration file. We will use the default "
                                         "configuration for this session."
                                         %(user_config_path, "%s.bak" % user_config_path))

                    # Move the config file to a backup file
                    shutil.copy2(user_config_path, "%s.bak" % user_config_path)

                    # Remove old file
                    os.remove(user_config_path)

                    # Copy the default configuration
                    shutil.copy2(self._default_path, user_config_path)

                else:

                    self._filename = user_config_path

                    print("Configuration read from %s" % (user_config_path))

        else:

            custom_warnings.warn("Using default configuration from %s. "
                                 "You might want to copy it to %s to customize it and avoid this warning."
                                 % (self._default_path, user_config_path))

            self._configuration = self._default_configuration
            self._filename = self._default_path

    def __getitem__(self, key):

        if key in self._configuration.keys():

            return self._configuration[key]

        else:

            raise ValueError("Configuration key %s does not exist in %s." % (key, self._filename))

    def __repr__(self):

        return yaml.dump(self._configuration, default_flow_style=False)

    def restore_default_configuration(self):
        """
        Restore the default configuration

        :return:

        """

        thisFilename = get_path_of_default_configuration()

        if os.path.exists(thisFilename):

            with open(thisFilename) as f:

                configuration = yaml.safe_load(f)

                # Test if the distribution configuration

                if self._check_configuration(configuration, f):

                    self._configuration = configuration

                    print("Default configuration read from %s" % (thisFilename))

                else:

                    raise ConfigurationFileCorrupt('Default configuration is corrupted')

    def restore_user_configuration(self):
        """
        Restore the user configuration if it exists.


        :return:
        """

        self.__init__()

    @staticmethod
    def is_matplotlib_cmap(cmap):

        try:

            plt.get_cmap(cmap)

            return True


        except:

            return False

    @staticmethod
    def is_matplotlib_color(color):
        # color_converter = colors.ColorConverter()

        try:

            return colors.is_color_like(color)

        except(ValueError):

            return False

    @staticmethod
    def is_bool(var):
        return type(var) == bool

    @staticmethod
    def is_string(var):

        return type(var) == str

    def _subs_values_with_none(self, d):
        """
        This remove all values from d and all nested dictionaries of d, substituing all values with None

        :param d: input dictionary
        :return: a copy of d with all values substituted with None
        """
        if isinstance(d, dict):

            return {k: self._subs_values_with_none(d[k]) for k in d}

        else:

            # Replace all non-dict values with None.
            return None

    def _check_same_structure(self, d1, d2):
        """
        Return True if d1 and d2 have the same keys structure (same set of keys, and all nested dictionaries have
        the same structure)

        :param d1: dictionary 1
        :param d2: dictionary 2
        :return: True or False
        """

        # This uses the fact that two dictionaries are equal if they have the same keys and the same values

        return self._subs_values_with_none(d1) == self._subs_values_with_none(d2)

    def _traverse_dict(self, d):

        for key in d:

            if isinstance(d[key], dict):

                for key, value in self._traverse_dict(d[key]):

                    yield key, value

            else:

                yield key, d[key]

    def _check_configuration(self, config_dict, config_path):
        """
        A routine to make sure that user specified configurations
        are indeed valid.

        :param config_dict: dictionary with configuration
        :param config_path: path from which the configuration has been read
        :return: None, but raises exceptions if errors are encountered
        """

        # First check that the provided configuration has the same structure of the default configuration
        # (if a default configuration has been loaded)

        if (self._default_configuration is not None) and \
                (not self._check_same_structure(config_dict, self._default_configuration)):

            # It does not, so of course is not valid (no need to check further)

            raise ConfigurationFileCorrupt("Config file %s has a different structure than the expected "
                                           "one." % config_path)

        else:

            # This will contain a copy of the configuration where all the element_types have been removed
            # (so that a key "background (color)" in the config_dict become simply "background" in the new
            # one
            new_configuration = {}

            # Make a dictionary of known checkers and what they apply to
            known_checkers = {'color': (self.is_matplotlib_color, 'a matplotlib color (name or html hex value)'),
                              'cmap': (self.is_matplotlib_cmap, 'a matplotlib color map (available: %s)' %
                                       ", ".join(plt.colormaps())),
                              'name': (self.is_string, "a valid name (string)"),
                              'switch': (self.is_bool, "one of yes, no, True, False")}

            # Now that we know that the provided configuration have the right structure, let's check that
            # each value is of the proper type

            for key, value in self._traverse_dict(config_dict):

                # Each key is in the form "element_name (element_type)", for example "background (color)"

                try:

                    element_name, element_type = re.findall("(.+) \((.+)\)", key)[0]

                except IndexError:

                    raise ConfigurationFileCorrupt("Cannot parse element '%s' in configuration file %s" %(key,
                                                                                                          config_path))

                if element_type in known_checkers:

                    checker, descr = known_checkers[element_type]

                    if not checker(value):

                        raise ValueError("Value %s for key %s in file %s is not %s" % (value, element_name,
                                                                                       config_path, descr))

                else:

                    raise ConfigurationFileCorrupt("Cannot understand element type %s for "
                                       "key %s in config file %s" % (element_type, key, config_path))

                # If we are here everything is fine
                new_configuration[element_name] = value

            # If we are here it means that all checks were successful
            return new_configuration


# Now read the config file, so it will be available as Config.c
threeML_config = Config()
