"""Database Warmer"""
import argparse
import time
import pandas as pd
from module.database.database import Database

datasets = {
    'discogs': 'datasets/groundtruth/discogs-2017.tsv',
    'lastfm': 'datasets/groundtruth/lastfm-2017.tsv',
    'tagtraum': 'datasets/groundtruth/tagtraum-2017.tsv',
}

def parse_args():
    """Parse and stores the provided terminal args

    Parameters
    ----------

    Returns
    -------
    args : Parsed args

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="Dataset to use",
                        default='lastfm', type=str)

    return parser.parse_args()

def warmer(dataset):
    """Fill the MongoDB with data

    Parameters
    ----------
    dataset: str
        String of the dataset to use

    Returns
    -------
    """
    start = time.time()
    f = datasets[dataset]
    master_record = pd.read_csv(f, sep='\t', keep_default_na=False)
    db = Database(database='genre_classifier', collection=dataset)
    for _, row in master_record.iterrows():
        doc = {
            'mbid': row['recordingmbid'],
            'release': row['releasegroupmbid'],
            'genres': get_labels(row)
        }
        _id = db.insert(doc)
        print("Insert mbid : "+ doc['mbid'] +" with id : "+_id)
    
    print("Took "+(time.time() - start)+" seconds")

def get_labels(record):
    """Returns the record labels in a document ready fashion

    Parameters
    ----------
    record: pandas.row

    Returns
    -------
    output: list
        List cointaining the parsed genres

    """
    output = []
    for label in [x for x in record[2:] if x]:
        consume_label(label.split('---'), output)
    return output

def consume_label(array, output):
    if len(array) == 1:
        return output.append({
            "name": array[0],
            "genres": []
        })
    for label in output:
        if label['name'] == array[0]:
            return consume_label(array[1:], label['genres'])
    
if __name__ == '__main__':
    args = parse_args()
    warmer(**vars(args))