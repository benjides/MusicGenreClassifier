""" Core module """
import logging
from module.logger import setup_logging
from module.config import Config
from module.genre_classifier import GenreClassifier

def main(args=None):
    """ Program bootstrap """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Program started")
    Config.load_config(args.config, args.output)

    if args.test:
        test()
    else:
        train()


def train():
    """ Train phase """

def test():
    """ Test phase """
