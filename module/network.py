import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model


class Network(object):
    """Artificial Neural Network Classsifier. """

    BATCH_SIZE = 32
    EPOCHS = 150
    VALIDATION_SPLIT = 0.1

    model = None

    def __init__(self):
        print("Init Network")
        self.model = Sequential()
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])


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

    def save_model(self, save_dir, model_name):
        """Saves the model to disk

        Parameters
        ----------
            save_dir: relative path to the file to save
            model_name: name of the model to save

        Returns
        -------
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, model_name + '.h5')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def load_model(self, save_dir, model_name):
        """Loads the model from disk

        Parameters
        ----------
            save_dir: relative path to the file to save
            model_name: name of the model to save

        Returns
        -------
        """
        self.model = load_model(os.path.join(save_dir, model_name + '.h5'))
