import os
import logging
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from module.bp_mll import bp_mll_loss

class Network(object):
    """Artificial Neural Network Classsifier. """

    BATCH_SIZE = 62
    EPOCHS = 350
    VALIDATION_SPLIT = 0.1

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
        model = Sequential()
        model.add(Dense(128, input_dim=x_dim, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='sigmoid',
                        kernel_initializer='glorot_uniform'))
        model.add(Dense(y_dim, activation='sigmoid',
                        kernel_initializer='glorot_uniform'))
        model.compile(loss=bp_mll_loss, optimizer='adagrad',
                      metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train):
        """Trains the model

        Fits the data using the established network architecture

        Parameters
        ----------
            x_train: processed data for this subset
            y_train: processed label for each x
        """
        self.model.fit(x_train, y_train,
                       batch_size=self.BATCH_SIZE,
                       epochs=self.EPOCHS,
                       validation_split=self.VALIDATION_SPLIT)

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
