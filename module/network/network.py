""" ANN module """
import logging
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from module.network.network_builder import builder
from module.network.bp_mll import bp_mll_loss
from module.network.hamming_distance import hamming_distance
from module.config import Config

class Network(object):
    """Artificial Neural Network Classsifier. """

    logger = logging.getLogger(__name__)

    def __init__(self):
        self.model = None

    def compile_model(self, x_dim, y_dim):
        """Compiles the model

        Compiles the model ready to be used for the training phase

        Parameters
        ----------
            x_dim: int
                dimension of the input
            y_dim: int
                dimension of the output (num of classes)
        Returns
        -------
        """
        self.model = builder(x_dim, y_dim)

    def train(self, name, training_generator, validation_generator, epochs, workers):
        """Trains the model

        Fits the data using a generator

        Parameters
        ----------
            name: String
                Model name to save the tensorboard logs
            training_generator: Sequence
                Generator yielding training examples
            validation_generator: Sequence
                Generator yielding validation examples
            epochs: int
                Number of epochs to do the training
            workers: int
                CPU workers to process the data on each epoch
        Returns
        -------
            history: Model
                Fitted model with the history training values
        """
        callbacks = [
            EarlyStopping(patience=5, monitor='loss'),
            TensorBoard(log_dir="logs/{}".format(name))
        ]
        return self.model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            use_multiprocessing=True,
            workers=workers,
            callbacks=callbacks
        )

    def evaluate(self, test_generator):
        """Evaluates the model

        Fits the data using a generator

        Parameters
        ----------
            test_generator: Sequence
                generator yielding test examples
        Returns
        -------
            evaluation_metrics: list
                List containing the values of the loss and the metrics selected during the compile phase for this generator
        """
        return self.model.evaluate_generator(test_generator)

    def classify(self, example):
        """Classifies an example and provides labels to it

        Parameters
        ----------
            example: np.array
                Example to be classified

        Returns
        -------
            probs: np.array 
                Obtained labels for the provided example. (Prediction prob per class)
        """
        return self.model.predict(example)

    def classify_generator(self, generator):
        """Classifies an example and provides labels to it

        Parameters
        ----------
            example: np.array
                Example to be classified

        Returns
        -------
            probs: np.array 
                Obtained labels for the provided example. (Prediction prob per class)
        """
        return self.model.predict_generator(generator)

    def save_model(self, model_name):
        """Saves the model to disk

        Parameters
        ----------
            model_name: str
                name of the model to save

        Returns
        -------
        """
        self.model.save(model_name + '.h5')

    def load_model(self, model_name):
        """Loads the model from disk

        Parameters
        ----------
            model_name: str
                name of the model to load

        Returns
        -------
        """
        custom_objects = {
            'bp_mll_loss': bp_mll_loss,
            'hamming_distance': hamming_distance
        }
        self.model = tf.keras.models.load_model(model_name + '.h5', custom_objects=custom_objects)
