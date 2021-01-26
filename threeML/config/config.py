from omegaconf import OmegaConf


from threeML.io.package_data import get_path_of_user_config


from .config_structure import Config
# Read the default Config
threeML_config = OmegaConf.structured(Config)

# now glob the config directory

for user_config in get_path_of_user_config().glob("*.yml"):

    _partial_conf = OmegaConf.load(user_config)
    
    threeML_config = OmegaConf.merge(threeML_config, user_config)
    



