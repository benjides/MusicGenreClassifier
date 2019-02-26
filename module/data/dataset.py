import csv
import random
from example import Example

class Dataset(object):
    """Dataset class. """
    root = 'datasets/groundtruth/'
    source = {
        'discogs': 'discogs-2017.tsv',
        'lastfm': 'lastfm-2017.tsv',
        'tagtraum': 'tagtraum-2017.tsv',
    }

    def __init__(self, source):
        self.dataset = []
        with open(self.root + self.source[source], 'r') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                self.dataset.append(Example(row[0], row[2:]))
        print("Dataset loaded")

    def get_training(self):
        """Get the whole dataset

        Parameters
        ----------

        Returns
        -------
            dataset: The training dataset
        """
        return self.dataset

    def get_random(self):
        """Returns a random training example

        Parameters
        ----------

        Returns
        -------
            example: Random example.
        """
        return random.choice(self.dataset)
