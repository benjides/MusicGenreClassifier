""" Core module """
import logging
from module.logger import setup_logging

def main(args):
    """ Program bootstrap """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Program started")
