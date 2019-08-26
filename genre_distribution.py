'''Database Warmer'''
import argparse
import pandas as pd
from module.config import Config
from module.database.aggregator import Aggregator
from module.database.database import Database
from module.plotter import genre_distribution

def parse_args():
    '''Parse and stores the provided terminal args

    Parameters
    ----------

    Returns
    -------
    args : Parsed args

    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset to use',
                        default='all', type=str)

    return parser.parse_args()

def get_distribution(dataset):
    db = Database(database=Config.get()['dataset']['database'], collection=dataset)
    agg = Aggregator()
    agg.unwind('$genres')
    agg.group('$genres.name', count={"$sum" : 1})
    agg.sort(_id=1)
    records = db.run_aggregate(agg)
    return pd.DataFrame.from_records(records)
    
if __name__ == '__main__':
    args = parse_args()
    Config.load_config()

    labels = [
        'discogs', 'lastfm', 'tagtraum'
    ]
    
    if args.dataset != 'all':
        labels = [args.dataset]
    
    distributions = [get_distribution(label) for label in labels]
    genre_distribution(distributions, labels)