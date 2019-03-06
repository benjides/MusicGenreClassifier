""" Dataset Module """
import csv
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

    def __init__(self, source, records=None):
        self.records = records
        self.logger.info('Reading %s data async', source)
        csvfile = open(self.root + self.source[source], 'r')
        next(csvfile)
        self.reader = csv.reader(csvfile, delimiter='\t')

    def __iter__(self):
        '''Generator function.'''
        i = self.reader
        if self.records:
            i = (next(self.reader) for _ in range(self.records))
        return (Example(row[0], row[2:]).get() for row in i)
