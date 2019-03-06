""" Example Loader Module """
import json
import numpy as np

class Example(object):
    """Example class. """

    root = 'datasets/train/'
    separator = '---'

    def __init__(self, mbid, labels):
        self.mbid = mbid
        self.labels = {}
        with open(self.root + mbid[:2] + '/' + mbid + '.json', 'r') as jsonfile:
            data = json.load(jsonfile)
            self.data = self.set_data(data)
            self.labels = self.set_labels(labels)

    def get(self):
        """Gets the example attrs

        Returns
        ----------
            example: tuple
                Tuple cointaining the data and the example labels
        """
        return (self.data, self.labels)

    def set_data(self, data):
        """Defines the data of the example

        Parameters
        ----------
            data: raw data from the training
        """
        return np.array(data["lowlevel"]["mfcc"]["cov"]).flatten()

    def set_labels(self, labels):
        """Defines the labels of the example as hierarchical dictionary

        Parameters
        ----------
            labels: string with the concatenated labels and sublabels
        """
        l = {}
        for label in labels:
            self.feed_label(label, l)
        return l

    def feed_label(self, label, labels):
        """Sets the labels recursively

        Parameters
        ----------
            label: single label from the the groundtruth
            labels: dictionary with the keys
        """
        if not label:
            return
        chunks = label.split(self.separator)
        if chunks[0] not in labels.keys():
            labels[chunks[0]] = {}
        if len(chunks) == 1:
            return
        return self.feed_label(self.separator.join(chunks[1:]), labels[chunks[0]])
