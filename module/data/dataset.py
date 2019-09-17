import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from module.data.data_generation import data_generation
from module.data.example import get_input


class Dataset(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, batch_size=32):
        'Initialization'
        self.dataframe = dataframe
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.dataframe['X_train'].shape[0] // self.batch_size

    def __getitem__(self, idx):
        'Generate one batch of data'
        batch = self.dataframe['X_train'][idx * self.batch_size:(idx + 1) * self.batch_size,:]
        labels = self.dataframe['Y_train'][idx * self.batch_size:(idx + 1) * self.batch_size,:]

        return (batch, labels)


    def on_epoch_end(self):
        # self.dataframe = self.dataframe.sample(frac=1)
        print("wo")
