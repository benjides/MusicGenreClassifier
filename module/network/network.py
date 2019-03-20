""" ANN module """
import logging
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from module.network.network_builder import builder
from module.network.bp_mll import bp_mll_loss
from module.config import Config

class Network(object):
    """Artificial Neural Network Classsifier. """

    logger = logging.getLogger(__name__)

    model = None

    def __init__(self):
        self.logger.info("Init network")

    def compile_model(self, x_dim, y_dim):
        """Compiles the model

        Compiles the model ready to be used for the training phase

        Parameters
        ----------
            x_dim: dimension of the input
            y_dim: dimension of the output (num of classes)
        """
        self.logger.info("Compiling model")
        self.model = builder(x_dim, y_dim)

    def train_model(self, x_train, y_train):
        """Trains the model

        Fits the data using the established network architecture

        Parameters
        ----------
            x_train: processed data for this subset
            y_train: processed label for each x
        """
        self.model.fit(x_train, y_train, **Config.get()['train'])

    def classify(self, example):
        """Classifies an example and provides labels to it

        Parameters
        ----------
            example: example to be classified

        Returns
        -------
            labels: obtained labels for the provided example.
        """
        return self.model.predict(example)

    def save_model(self, model_name):
        """Saves the model to disk

        Parameters
        ----------
            model_name: name of the model to save

        Returns
        -------
        """
        self.model.save(model_name + '.h5')
        self.logger.info('Saved trained model at %s ', model_name)

    def load_model(self, model_name):
        """Loads the model from disk

        Parameters
        ----------
            model_name: name of the model to load

        Returns
        -------
        """
        self.model = load_model(model_name + '.h5', custom_objects={'bp_mll_loss': bp_mll_loss})
