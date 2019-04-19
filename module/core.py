""" Core module """
import logging
import numpy as np
import tensorflow as tf
from module.logger import setup_logging
from module.config import Config
from module.genre_classifier import GenreClassifier
from module.database.database import Database
from module.data.samples import get_random

def main(args=None):
    """ Program bootstrap """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Program started")
    Config.load_config(output=args.output)
    tf.enable_eager_execution()

    classifier = GenreClassifier()
    if args.test:
        test(classifier)
    else:
        train(classifier)


def train(classifier):
    """ Train phase """
    classifier.train_model()

def test(classifier):
    """ Test phase """
    db = Database(Config.get()['dataset']['database'], Config.get()['dataset']['source'])
    samples = get_random(db, classifier.path, classifier.genre, Config.get()['dataset']['records'])
    classifier.evaluate_model(samples)
