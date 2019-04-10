import pandas as pd
import numpy as np
from math import ceil
from keras.utils import Sequence
from module.data.example import get_input

class Dataset(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, genre, batch_size=32):
        'Initialization'
        self.dataframe = dataframe
        self.genre = genre
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
            
            data = get_input(row['mbid'])
            label = 1 if row['genre'] == self.genre else 0

            batch.append(data)
            labels.append(label)

        batch = np.array(batch)
        return (batch, labels)
