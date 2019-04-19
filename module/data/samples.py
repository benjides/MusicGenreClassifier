import pandas as pd
import json
from module.database.aggregator import Aggregator

def get_samples(db, path, genre):
    query = Aggregator()

    query.group(
        "$release",
        mbid={
            "$first": "$mbid"
        },
        genres={
            "$first": "$genres"
        }
    )

    agg = "$genres"
    for _ in range(path.count('.')):
        query.unwind(agg)
        agg += ".genres"

    if genre is not None:
        query.match(path, genre)
        
    query.project(
        _id=0,
        mbid=1,
        genre="$genres."+path
    )
    
    documents = db.run_aggregate(query)
    return pd.DataFrame.from_records(documents)

def get_random(db, path, genre, records):
    return get_samples(db, path, genre).sample(records)
