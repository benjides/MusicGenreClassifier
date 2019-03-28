"""Genre classifier Module"""
import os
import logging
import dill
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from module.config import Config
from module.network.network import Network
from module.data.example import get_example

class GenreClassifier(object):
    """Class to define a hierachical genre classifier. """

    logger = logging.getLogger(__name__)

    constants = {
        'models': 'models/',
        'data': {
            'discogs': 'datasets/groundtruth/discogs-2017.tsv',
            'lastfm': 'datasets/groundtruth/lastfm-2017.tsv',
            'tagtraum': 'datasets/groundtruth/tagtraum-2017.tsv',
        }
    }

    def __init__(self, dataframe=None, genre='main', index=0):
        self.genre = genre
        self.index = index
        self.dataframe = self.load_dataframe(dataframe)
        self.labels = self.load_labels()
        self.load_model()

    def train_model(self):
        """Trains the model

        Trains the model and triggers the consequent classifiers hierarchically

        Parameters
        ----------
            data: raw data for this subset
        """
        data, _ = get_example(self.dataframe.iloc[0])
        network = Network()
        network.compile_model(data.shape[0], len(self.labels.classes_))
        steps_per_epoch = len(self.dataframe.index) // Config.get()['train']['batch_size']
        network.train_generator(self.generator(), steps_per_epoch)
        self.save_model(network)
        # for each sublabel in labels
            # divided_data = self.divide_data(data, label)
            # self.classifiers[label] = GenreClassifier(label, self.index + 1)
            # self.classifiers[label].train(divided_data)

    def classify(self, batch):
        """Classifies a batch of examples and provides labels to it

        Parameters
        ----------
            batch: examples to be classified

        Returns
        -------
            labels: obtained labels for the provided batch.
        """
        network = Network()
        network.load_model(self.get_model_name())
        result = network.classify(batch)
        return self.labels.inverse_transform(result)
        # TODO
        # labels = Classify it using the model
        # For each obtained label in labels
            # result[label] = self.classifier[label].classify(example)
        # return labels

    def save_model(self, network):
        """Save the computed model

        saves this object and the computed model to disk to quick reload

        Parameters
        ----------
            network: trained network

        """
        model = self.get_model_name()
        dirname = os.path.dirname(model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(model + '.pkl', 'wb') as f:
            dill.dump(self, f, protocol=-1)
            network.save_model(model)

    def load_dataframe(self, dataframe):
        if dataframe is not None:
            return dataframe

        f = self.constants['data'][Config.get()['dataset']['source']]
        return pd.read_csv(f, sep='\t', keep_default_na=False)

    def load_labels(self):
        labels = self.dataframe['genre1'].unique()
        mlb = LabelBinarizer()
        return mlb.fit(labels)

    def load_model(self):
        """Loads the model

        Attempts to load this model (ONLY THIS OBJECT, the network is loaded somewhere else)
        """
        model = self.get_model_name()+'.pkl'
        if not os.path.exists(model):
            return
        with open(model, "rb") as f:
            model = dill.load(f)
            self.labels = model.labels
            self.dataframe = model.dataframe

    def get_model_name(self):
        """Gets the model name

        Obtains the mode name withouth the extension
        to be used to store the GenreClassifier and Network objects

        Returns
        -------
            model: string
                Model name
        """
        return self.constants['models']+Config.get()['output']+'_'+self.genre+str(self.index)

    def generator(self):
        '''Generator function.'''
        rows = self.dataframe.iterrows()
        batch_size = Config.get()['train']['batch_size']
        while True:
            batch, labels = [], []
            iterator = (next(rows) for _ in range(batch_size))
            for _, row in iterator:
                data, label = get_example(row)
                #Discard unwanted genres and refine

                batch.append(data)
                labels.append(list(label.keys())[0])

            batch = np.array(batch)
            labels = self.labels.transform(labels)
            yield (batch, labels)
