import csv
import random
import logging
from module.data.example import Example

class Dataset(object):
    """Dataset class. """
    root = 'datasets/groundtruth/'
    source = {
        'discogs': 'discogs-2017.tsv',
        'lastfm': 'lastfm-2017.tsv',
        'tagtraum': 'tagtraum-2017.tsv',
    }
    logger = logging.getLogger(__name__)

    def __init__(self, source):
        self.logger.info('Reading %s data async', source)
        csvfile = open(self.root + self.source[source], 'r')
        next(csvfile)
        self.reader = csv.reader(csvfile, delimiter='\t')

    def __iter__(self):
        '''Generator function.'''
        for row in self.reader:
            yield Example(row[0], row[2:])
