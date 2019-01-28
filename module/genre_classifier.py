"""Genre classifier Module"""

from base_classifier import BaseClassifier

class GenreClassifier(BaseClassifier):
    """Class to define a hierachical genre classifier. """

    def __init__(self, genre=None, index=0):
        BaseClassifier.__init__(self, genre + index)

    def train_model(self, data):
        """Trains the model

        Trains the model and triggers the consequent classifiers hierarchically

        Parameters
        ----------
            data: processed data for this subset
        """
        # TODO
        # refined_data = self.refine_data(data)
        # self.train(refined_data)
        # for each sublabel in labels
            # divided_data = self.divide_data(data, label)
            # self.classifiers[label] = GenreClassifier(label, self.index + 1)
            # self.classifiers[label].train(divided_data)
        raise NotImplementedError

    def classify(self, example):
        """Classifies an example and provides labels to it

        Parameters
        ----------
            example: example to be classified

        Returns
        -------
            labels: obtained labels for the provided example.
        """
        # TODO
        # labels = Classify it using the model
        # For each obtained label in labels
            # result[label] = self.classifier[label].classify(example)
        # return labels
        raise NotImplementedError

    def refine_data(self, data):
        """Takes the provided data and adjusts it to the classifier standards

        Parameters
        ----------
            data: raw data from upper level

        Returns
        -------
            data: refined data ready to be processed by the classifier
        """
        raise NotImplementedError

    def divide_data(self, data, label):
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
        # TODO
        # divided_data = []
        # for each example in data
            # if example[label]
                # divided_data.push(example)
        raise NotImplementedError
