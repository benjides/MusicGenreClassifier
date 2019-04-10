"""Database Module Module"""
from pymongo import MongoClient

class Database(object):
    'Initializes the connection'
    def __init__(self, database, collection, host='localhost', port=27017):
        client = MongoClient(host, port)
        self.db = client[database]
        self.collection = self.load_collection(collection)

    def load_collection(self, collection):
        """Loads the collection

        Creates (if not exist) the desired collection and loads it to further use

        Parameters
        ----------
            collection: str
                collection to use
        Returns
        -------
        """
        if collection in self.db.list_collection_names():
            return self.db[collection]
        self.db.create_collection(collection)
        coll = self.db[collection]
        coll.create_index("mbid", unique=True)
        return coll


    def insert(self, document):
        """Inserts a document into the collection

        Parameters
        ----------
            document: dict(bson)
                document to insert
        Returns
        -------
            inserted_id: str
                string representation of the inserted id
        """
        return self.collection.insert_one(document).inserted_id

    def insert_many(self, documents):
        """Inserts many documents into the collection

        Parameters
        ----------
            documents: list(dict(bson))
                documents to insert
        Returns
        -------
            inserted_ids: list(str)
                string representation of the inserted ids
        """
        return self.collection.insert_many(documents).inserted_ids

    
    def run_aggregate(self, pipeline):
        """Runs the query pipeline

        Parameters
        ----------
            pipeline: list(dict())
                List containing the pipeline stages in MQL
        Returns
        -------
            aggregate: cursor
                MongoDB cursor object
        """
        return self.collection.aggregate(pipeline)