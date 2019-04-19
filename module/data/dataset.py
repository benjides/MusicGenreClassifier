import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from module.data.data_generation import data_generation
from module.data.example import get_input


class Dataset(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, mlb, batch_size=32):
        'Initialization'
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.mlb = mlb

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataframe.index) // self.batch_size

    def __getitem__(self, idx):
        'Generate one batch of data'
        rows = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        return data_generation(rows, self.mlb)
