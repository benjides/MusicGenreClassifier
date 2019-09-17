"""Genre classifier Module"""
import os
import logging
import dill
import json
import pandas as pd
import numpy as np 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, binarize
from module.config import Config
from module.data.dataset import Dataset
from module.data.data_generation import data_generation
from module.data.example import get_input
from module.data.genres import get_genres
from module.data.samples import get_samples
from module.data.split_dataset import split_dataset
from module.database.aggregator import Aggregator
from module.database.database import Database
from module.network.network import Network

class GenreClassifier(object):
    """Class to define a hierachical genre classifier. """

    logger = logging.getLogger(__name__)
    models = 'models/'
    reports = 'reports/'

    def __init__(self, path='name', genre=None, index=0):
        self.path = path
        self.genre = genre
        self.index = index
        self.genres = self.get_genres()
        self.mlb = self.set_mlb()

    def train_model(self):
        """Trains the model

        Trains the model and triggers the consequent classifiers hierarchically

        Parameters
        ----------
        Returns
        -------
        """
        #if self.genre is not None:
        self.train()
        
        # for genre in self.genres:
        #     g = GenreClassifier(
        #         path='genres.'+self.path,
        #         genre=genre['_id'],
        #         index=self.index + 1
        #     )
        #     g.train_model()

    def train(self):
        """Trains the model

        Trains the model

        Parameters
        ----------
        Returns
        -------
        """

        yeast = np.load('yeast.npz')

        X_train = yeast['X_train']
        Y_train = yeast['Y_train']

        n = X_train.shape[0]
        dim_no = X_train.shape[1]
        class_no = Y_train.shape[1]
       
        network = Network()
        self.logger.info("Compiling Model")
        network.compile_model(dim_no, class_no)


        training_generator = Dataset(
            yeast,
            batch_size=Config.get()['train']['batch_size'],
        )

        self.logger.info('Training Model : %s ', self.get_model_name())

        network.train(
            name=self.get_model_name(),
            training_generator=training_generator,
            epochs=Config.get()['train']['epochs'],
            workers=Config.get()['train']['workers']
        )

        self.save_model(network)

        self.evaluate_model(test)

    def evaluate_model(self, dataframe):
        x, y_true = data_generation(dataframe, self.mlb)
        y_pred = self.classify(x)
        y_pred = binarize(y_pred, Config.get()['dataset']['threshold']).astype(int)
        model = self.get_model_name()
        with open(self.reports + model + '.txt', 'w') as f:
            f.write("Accuracy (train) for %s: %0.1f%% \n" % (model, accuracy_score(y_true, y_pred) * 100))
            # f.write(confusion_matrix(y_true, y_pred)+"\n")
            f.write(classification_report(y_true, y_pred)+"\n")

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
        network.load_model(self.get_model_path())
        return binarize(network.classify(batch), Config.get()['dataset']['threshold'])
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
        model = self.get_model_path()
        dirname = os.path.dirname(model)
        network.save_model(model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(model + '.pkl', 'wb') as f:
            dill.dump(self, f, protocol=-1)

        with open(model + '.json', 'w') as f:
            json.dump(Config.get(), f, indent=4)

        network.save_model(model)
        
        self.logger.info('Saved trained model at %s ', model)

    def load_dataframe(self):
        db = Database(Config.get()['dataset']['database'], Config.get()['dataset']['source'])
        return get_samples(db, self.path, self.genre)

    def get_genres(self):
        db = Database(Config.get()['dataset']['database'], Config.get()['dataset']['source'])
        return get_genres(db, self.path, self.genre)

    def set_mlb(self):
        labels = [[label['_id']] for label in self.genres]
        mlb = MultiLabelBinarizer()
        return mlb.fit(labels)

    def load_model(self):
        """Loads the model

        Attempts to load this model (ONLY THIS OBJECT, the network is loaded somewhere else)
        """
        model = self.get_model_path()+'.pkl'
        if not os.path.exists(model):
            return
        with open(model, "rb") as f:
            model = dill.load(f)
            self.path = model.path
            self.genre = model.genre
            self.index = model.index
            self.genres = model.genres

    def get_model_path(self):
        """Gets the model path+name

        Obtains the model as a path name withouth the extension
        to be used to store the GenreClassifier and Network objects

        Returns
        -------
            model: string
                Model name
        """
        return self.models+self.get_model_name()

    def get_model_name(self):
        """Gets the model name

        Obtains the mode name withouth the extension
        to be used to store the GenreClassifier and Network objects

        Returns
        -------
            model: string
                Model name
        """
        return Config.get()['output']+'_'+(self.genre or 'main')+str(self.index)