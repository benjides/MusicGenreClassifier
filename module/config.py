""" Config Module """
import json

class Config(object):
    """Config static class"""

    config = {}

    @staticmethod
    def load_config(filename='config.example'):
        """Config loader

        Loads the config file and makes it available to the whole project

        Parameters
        ----------
            filename: configuration file (withouth the extension)

        Returns
        -------
        """
        with open(filename + '.json') as f:
            config = json.load(f)
            config['output'] = filename
            Config.config = config

    @staticmethod
    def get():
        """Gets the config dict

        Parameters
        ----------

        Returns
        -------
            config : the config dictionary
        """
        if not any(Config.config):
            Config.load_config()
        return Config.config
