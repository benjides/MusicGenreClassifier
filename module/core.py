""" Core module """
import logging
from module.logger import setup_logging
from module.config import Config

def main(args=None):
    """ Program bootstrap """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Program started")
    Config.load_config(args.config)
