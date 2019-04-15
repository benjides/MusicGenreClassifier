"""Database Module Module"""
from pymongo import MongoClient
from module.database.aggregator import Aggregator

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

    def set_collection(self, collection):
        """Sets the collection to use

        Parameters
        ----------
            collection: str
                collection to use
        Returns
        -------
        """
        self.collection = self.load_collection(collection)

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

    def run_aggregate(self, aggregator):
        """Builds and runs the Aggregator query

        Parameters
        ----------
            pipeline: Aggregator
                Aggregator query
        Returns
        -------
            cursor: CommandCursor
                MongoDB cursor containing the results
        """
        return self.run_pipeline(aggregator.build())

    def run_pipeline(self, pipeline):
        """Runs the query pipeline

        Parameters
        ----------
            pipeline: list(dict())|Aggregator
                documents to insert
        Returns
        -------
            cursor: CommandCursor
                MongoDB cursor containing the results
        """
        return self.collection.aggregate(pipeline)