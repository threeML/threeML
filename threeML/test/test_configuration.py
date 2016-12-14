from threeML.config.config import Config


def test_default_configuration():

    # We just need to instance the Config class, as it contains in itself the check for a valid
    # default configuration file (it will raise an exception if the file is not valid)

    c = Config()
