import pandas as pd
from module.database.aggregator import Aggregator

def get_samples(db, path, genre):

    query = Aggregator() \
        .match(path, genre) \
        .group("$release", mbid={"$first": "$mbid"}) \
        .project(_id=0, mbid=1, genre='1')
    
    positives = list(db.run_aggregate(query))

    query = Aggregator() \
        .match("genres.name", {'$ne': genre}) \
        .group("$release", mbid={"$first": "$mbid"}) \
        .project(_id=0, mbid=1, genre='0') \
        .limit(len(positives))

    negatives = list(db.run_aggregate(query))

    return pd.DataFrame(positives + negatives)

def get_random(db, path, genre, records):
    
    return get_samples(db, path, genre).sample(records)
