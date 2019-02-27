import logging
from module.logger import setup_logging

def main(args = {}):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Program started")