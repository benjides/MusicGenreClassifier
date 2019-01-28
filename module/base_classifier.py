"""BaseClassifier Module"""

from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    """Class to define the basic functionality of a classifier. """

    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, data):
        """Trains a model and saves it to disc

        Parameters
        ----------
            data: data to be used during the training phase

        Returns
        -------
        """
        self.train_model(data)
        self.save()

    @abstractmethod
    def train_model(self, data):
        """Trains the model using the provided data

        Parameters
        ----------
            data: data to be used during the training phase

        Returns
        -------
        """
        raise NotImplementedError

    @abstractmethod
    def classify(self, example):
        """Classifies an example and provides and label to it

        Parameters
        ----------
            example: example to be classified

        Returns
        -------
            label: label of the example
        """
        raise NotImplementedError

    def save(self):
        """Saves the model to the disc

        Parameters
        ----------

        Returns
        -------
        """
        pass

    def load(self):
        """Loads the model fromthe disc

        Parameters
        ----------

        Returns
        -------
        """
        pass

    def delete(self):
        """Deletes the model from the disc

        Parameters
        ----------

        Returns
        -------
        """
        pass
