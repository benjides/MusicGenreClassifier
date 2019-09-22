import pandas as pd
import json
from module.database.aggregator import Aggregator

def get_genres(db, path, genre):

    query = Aggregator()

    agg = "$genres"
    for _ in path.split('.'):
        query.unwind(agg)
        agg += ".genres"

    if genre is not None:
        query.match(path, genre)

    query.group("$genres."+path, count={"$sum" : 1})
    query.match("count", {"$gt" : 4000})
    query.sort(count=1) 
    return list(db.run_aggregate(query))

