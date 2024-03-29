import numpy as np
from module.data.example import get_input

def data_generation(rows):
        'Generates data containing batch_size samples'
        batch, labels = [], []
        for _, row in rows.iterrows():
            
            data = get_input(row['mbid'])
            label = row['genre']

            batch.append(data)
            labels.append(label)

        batch = np.array(batch)
        labels = np.array(labels, dtype=int)
        return (batch, labels)