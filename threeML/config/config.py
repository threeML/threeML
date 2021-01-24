from __future__ import print_function

import os
import re
import shutil
import urllib.parse
from builtins import object
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pkg_resources
import yaml
from future import standard_library

from threeML.exceptions.custom_exceptions import (ConfigurationFileCorrupt,
                                                  custom_warnings)
from threeML.io.package_data import get_path_of_data_file, get_path_of_user_dir

standard_library.install_aliases()


_config_file_name = "threeML_config.yml"

# Scipy optimizers
# adds the ability for safe load to import dictionaries
_optimize_methods = (
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "dogleg",
    "trust-ncg",
)


class Config(object):
    def __init__(self):

        # Read first the default configuration file
        default_configuration_path: Path = get_path_of_data_file(
            _config_file_name)

        assert (
            default_configuration_path.exists()
        ), f"Default configuration {default_configuration_path} does not exist. Re-install 3ML"

        with default_configuration_path.open() as f:

            try:

                configuration: dict = yaml.load(f, Loader=yaml.FullLoader)

            except:

                raise ConfigurationFileCorrupt(
                    f"Default configuration file {default_configuration_path} cannot be parsed!"
                )

            # This needs to be here for the _check_configuration to work

            self._default_configuration_raw = configuration

            # Test the default configuration

            try:

                self._check_configuration(
                    configuration, default_configuration_path)

            except:

                raise

            else:

                self._default_path: Path = default_configuration_path

        # Check if the user has a user-supplied config file under .threeML

        user_config_path: Path = get_path_of_user_dir() / _config_file_name

        if user_config_path.exists():

            with user_config_path.open() as f:

                configuration: dict = yaml.load(f, Loader=yaml.FullLoader)

                # Test if the local/configuration is ok

                try:

                    self._configuration = self._check_configuration(
                        configuration, user_config_path
                    )

                except ConfigurationFileCorrupt:

                    # Probably an old configuration file
                    custom_warnings.warn(
                        f"The user configuration file at {user_config_path} does not appear to be valid. We will "
                        "substitute it with the default configuration. You will find a copy of the "
                        f"old configuration at {user_config_path}.bak so you can transfer any customization you might "
                        "have from there to the new configuration file. We will use the default "
                        "configuration for this session."
                    )

                    self.copy_default_config_file()

                else:

                    self._filename: Path = user_config_path

                    print(f"Configuration read from {user_config_path}")

        else:

            custom_warnings.warn(
                f"Using default configuration from {self._default_path}.\n"
                f"You might want to copy it to {user_config_path} to customize it and avoid this warning.\n"
                "You can also call threeML_config.copy_default_cong_file()\n"
            )

            self._configuration = self._check_configuration(
                self._default_configuration_raw, self._default_path
            )
            self._filename: Path = self._default_path

    def copy_default_config_file(self):

        user_config_path: Path = get_path_of_user_dir() / _config_file_name

        try:
            old_config = user_config_path.rename(f"{user_config_path}.bak")

            # Remove old file
            user_config_path.unlink()

        except:

            pass

            # Copy the default configuration
        shutil.copy(self._default_path, user_config_path)

        self._configuration = self._check_configuration(
            self._default_configuration_raw, self._default_path
        )
        self._filename: Path = self._default_path

    def __getitem__(self, key):

        if key in list(self._configuration.keys()):

            return self._configuration[key]

        else:

            raise ValueError(
                f"Configuration key {key} does not exist in {self._filename}"
            )

    def __repr__(self):

        return yaml.dump(self._configuration, default_flow_style=False)

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

        except (ValueError):

            return False

    @staticmethod
    def is_bool(var) -> bool:
        return type(var) == bool

    @staticmethod
    def is_string(var) -> bool:

        return type(var) == str

    @staticmethod
    def is_ftp_url(var) -> bool:

        try:

            tokens = urllib.parse.urlparse(var)

        except:

            # This is very rare, as almost anything is a valid URL
            return False

        else:

            if tokens.scheme != "ftp" or tokens.netloc == "":

                return False

            else:

                return True

    @staticmethod
    def is_http_url(var) -> bool:

        try:

            tokens = urllib.parse.urlparse(var)

        except:

            # This is very rare, as almost anything is a valid URL
            return False

        else:

            if (
                tokens.scheme != "http" and tokens.scheme != "https"
            ) or tokens.netloc == "":

                return False

            else:

                return True

    @staticmethod
    def is_optimizer(method) -> bool:

        if method in _optimize_methods:

            return True

        else:

            return False

    @staticmethod
    def is_path(path) -> bool:

        try:

            Path(path)
            return True
        except:

            return False

    @staticmethod
    def is_number(val) -> bool:

        return type(val) == int or type(val) == float

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

    def _check_same_structure(self, d1, d2) -> bool:
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

    def _check_configuration(self, config_dict: dict, config_path: Path) -> None:
        """
        A routine to make sure that user specified configurations
        are indeed valid.

        :param config_dict: dictionary with configuration
        :param config_path: path from which the configuration has been read
        :return: None, but raises exceptions if errors are encountered
        """

        # First check that the provided configuration has the same structure of the default configuration
        # (if a default configuration has been loaded)

        if (self._default_configuration_raw is not None) and (
            not self._check_same_structure(
                config_dict, self._default_configuration_raw)
        ):

            # It does not, so of course is not valid (no need to check further)

            raise ConfigurationFileCorrupt(
                f"Config file {config_path} has a different structure than the expected "
                "one."
            )

        else:

            # Make a dictionary of known checkers and what they apply to
            known_checkers = {
                "color": (
                    self.is_matplotlib_color,
                    "a matplotlib color (name or html hex value)",
                ),
                "cmap": (
                    self.is_matplotlib_cmap,
                    "a matplotlib color map (available: %s)"
                    % ", ".join(plt.colormaps()),
                ),
                "name": (self.is_string, "a valid name (string)"),
                "switch": (self.is_bool, "one of yes, no, True, False"),
                "ftp url": (self.is_ftp_url, "a valid FTP URL"),
                "http url": (self.is_http_url, "a valid HTTP(S) URL"),
                "optimizer": (
                    self.is_optimizer,
                    "one of scipy.optimize minimization methods (available: %s)"
                    % ", ".join(_optimize_methods),
                ),
                "number": (self.is_number, "an int or float"),
                "path": (self.is_path, "a path")
            }

            # Now that we know that the provided configuration have the right structure, let's check that
            # each value is of the proper type

            for key, value in self._traverse_dict(config_dict):

                # Each key is in the form "element_name (element_type)", for example "background (color)"

                try:

                    element_name, element_type = re.findall(
                        "(.+) \((.+)\)", key)[0]

                except IndexError:

                    raise ConfigurationFileCorrupt(
                        "Cannot parse element '%s' in configuration file %s"
                        % (key, config_path)
                    )

                if element_type in known_checkers:

                    checker, descr = known_checkers[element_type]

                    if not checker(value):

                        raise ValueError(
                            "Value %s for key %s in file %s is not %s"
                            % (value, element_name, config_path, descr)
                        )

                else:

                    raise ConfigurationFileCorrupt(
                        "Cannot understand element type %s for "
                        "key %s in config file %s" % (
                            element_type, key, config_path)
                    )

            # If we are here it means that all checks were successful
            # Return the new configuration, where all types are stripped out
            return self._get_copy_with_no_types(config_dict)

    @staticmethod
    def _remove_type(d) -> dict:

        # tmp = [ (key.split("(")[0].rstrip(), value) for key, value in d.items()]

        return dict((key.split("(")[0].rstrip(), value) for key, value in d.items())

    def _get_copy_with_no_types(self, multilevelDict):

        new = self._remove_type(multilevelDict)

        for key, value in new.items():

            if isinstance(value, dict):

                new[key] = self._get_copy_with_no_types(value)

            else:

                # Sometimes the user uses 'None' instead of None, which becomes the string
                # 'None' instead of the object None. Let's fix transparently this
                if new[key] == "None":

                    new[key] = None

        return new


# Now read the config file, so it will be available as Config.c
threeML_config = Config()
