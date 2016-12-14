from threeML.config.config_checker import check_configuration
import pkg_resources
import yaml
import os


def test_default_configuration():
    # check that the built in configuration file is not corrupt
    # i.e. that it has the right keys and the keys make sense

    distribution = pkg_resources.get_distribution("threeML")
    distribution_path = os.path.join(distribution.location, 'threeML/config')

    thisFilename = os.path.join(distribution_path, 'threeML_config.yml')

    if os.path.exists(thisFilename):

        with open(thisFilename) as f:

            configuration = yaml.safe_load(f)

            assert check_configuration(configuration, f) == True
