"""Agregator Builder"""
import copy
class Aggregator(object):

    'Initializes the builder'
    def __init__(self):
        self.pipeline = []

    def match(self, field, expression):
        """Filters the documents matching the specified condition(s)

        Parameters
        ----------
            field: str
                Document field

            expresion: str|dict
                Expression in MQL
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$match': {
                    field: expression
                }
            }
        )
        return self

    def group(self, _id, **kwargs):
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
        kwargs['_id'] = _id
        self.pipeline.append(
            {
                '$group': kwargs
            }
        )
        return self

    def limit(self, documents):
        """Provide the number of documents to limit.

        Parameters
        ----------
            records: int
                documents to get
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$limit': documents
            }
        )
        return self

    def project(self, **kwargs):
        """Passes the requested fields to the next stage in the pipeline.

        Parameters
        ----------
            kwargs: dict
                projection in the form { <field1>: <true|false>, <field2>: <true|false> ... } }
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$project': kwargs
            }
        )
        return self

    def sort(self, **kwargs):
        """Provide any number of field/order pairs.

        Parameters
        ----------
            kwargs: dict
                sorting in the form { <field1>: <sort order>, <field2>: <sort order> ... } }
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$sort': kwargs
            }
        )
        return self
    
    def unwind(self, path):
        """Deconstructs an array field and outputs a document for each element.

        Parameters
        ----------
            path: str
                field path
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$unwind': path
            }
        )
        return self

    def sample(self, documents):
        """Provide the number of documents to limit.

        Parameters
        ----------
            documents: int
                documents to sample
        Returns
        -------
            self: Aggregator
                Aggregator builder interface
        """
        self.pipeline.append(
            {
                '$sample': {
                    'size': documents
                }
            }
        )
        return self

    def lookup(self, join, localField, foreignField, field):
        self.pipeline.append(
            {
                '$lookup': {
                    'from': join,
                    'localField': localField,
                    'foreignField': foreignField,
                    'as': field
                }
            }
        )
        return self

    def out(self, collection):
        self.pipeline.append(
            {
                '$out': collection
            }
        )
        return self

    
    def build(self):
        """Returns the pipeline ready to be ran

        Parameters
        ----------
        Returns
        -------
            pipeline: list(dict)
                Built pipeline
        """
        return self.pipeline

    def clone(self):
        agg = Aggregator()
        agg.pipeline = copy.deepcopy(self.pipeline)
        return agg