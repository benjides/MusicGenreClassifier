"""Genre classifier Module"""
import os
import logging
import dill
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from module.network import Network

class GenreClassifier(object):
    """Class to define a hierachical genre classifier. """

    logger = logging.getLogger(__name__)
    root = 'models/'

    def __init__(self, genre='main', index=0):
        self.genre = genre
        self.index = index
        self.labels = []
        self.load_model()

    def train_model(self, data):
        """Trains the model

        Trains the model and triggers the consequent classifiers hierarchically

        Parameters
        ----------
            data: processed data for this subset
        """
        refined_data = self.refine_data(data)
        self.train(refined_data)
        # for each sublabel in labels
            # divided_data = self.divide_data(data, label)
            # self.classifiers[label] = GenreClassifier(label, self.index + 1)
            # self.classifiers[label].train(divided_data)

    def train(self, data):
        """Trains the network

        Trains the neural network

        Parameters
        ----------
            data: processed data for this subset
        """
        x_train, y_train = [], []
        self.logger.info('Buffering data for %s at depth %i', self.genre, self.index)
        for example, label in data:
            x_train.append(example)
            y_train.append(label)

        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        self.labels = list(mlb.classes_)
        x_train = np.array(x_train)
        self.logger.info('Data ready')
        network = Network()
        network.compile_model(x_train.shape[1], y_train.shape[1])
        network.train_model(np.array(x_train), y_train)
        self.save_model(network)

    def classify(self, example):
        """Classifies an example and provides labels to it

        Parameters
        ----------
            example: example to be classified

        Returns
        -------
            labels: obtained labels for the provided example.
        """
        network = Network()
        network.load_model(self.root+self.genre+str(self.index))
        return network.classify(example)
        # TODO
        # labels = Classify it using the model
        # For each obtained label in labels
            # result[label] = self.classifier[label].classify(example)
        # return labels

    def refine_data(self, dataset):
        """Takes the provided data and adjusts it to the classifier standards

        Parameters
        ----------
            data: raw data from upper level

        Returns
        -------
            dataset: refined data ready to be processed by the classifier
        """
        # return ((data, list(label.keys())) for data, label in dataset)
        for data, label in dataset:
            yield (data, list(label.keys()))

    def divide_data(self, dataset, label):
        """Divides the data according to the provided label

        Splits the data and divides it
        to only take into account the examles which have the desired label in the hierarchy

        Parameters
        ----------
            data: raw data from upper level
            label: label to take into account

        Returns
        -------
            data: divided data ready to be fed to the next classifier
        """
        for data, labels in dataset:
            if labels[label]:
                yield (data, labels[label])

    def save_model(self, network):
        """Save the computed model

        saves this object and the computed model to disk to quick reload

        Parameters
        ----------
            network: trained network

        """
        model = self.root+self.genre+str(self.index)
        dirname = os.path.dirname(model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(model + '.pkl', 'wb') as f:
            dill.dump(self, f, protocol=-1)
            network.save_model(model)

    def load_model(self):
        """Loads the model

        Attempts to load this model (ONLY THIS OBJECT, the network is loaded somewhere else)
        """
        model = self.root+self.genre+str(self.index)+'.pkl'
        if not os.path.exists(model):
            return
        with open(model, "rb") as f:
            model = dill.load(f)
            self.labels = model.labels
