import pandas as pd
import numpy as np
from math import ceil
from tensorflow.keras.utils import Sequence
from module.data.example import get_example

class Dataset(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, mlb, batch_size=32,):
        'Initialization'
        self.dataframe = dataframe
        self.mlb = mlb
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return ceil(len(self.dataframe.index) / self.batch_size)

    def __getitem__(self, idx):
        'Generate one batch of data'
        rows = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.__data_generation(rows)


    def __data_generation(self, rows):
        'Generates data containing batch_size samples'
        batch, labels = [], []
        for _, row in rows.iterrows():
            data, label = get_example(row)
            #Discard unwanted genres and refine

            batch.append(data)
            labels.append(list(label.keys()))

        batch = np.array(batch)
        labels = self.mlb.transform(labels)
        return (batch, labels)
