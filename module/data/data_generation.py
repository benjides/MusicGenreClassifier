import numpy as np
from module.data.example import get_input
from sklearn.preprocessing import StandardScaler

def data_generation(rows, mlb):
        'Generates data containing batch_size samples'
        scaler = StandardScaler()
        batch, labels = [], []
        for _, row in rows.iterrows():
            
            data = get_input(row['mbid'])
            label = row['genre']

            batch.append(data)
            labels.append(label)

        batch = np.array(batch)
        batch = scaler.fit_transform(batch.T).T
        labels = mlb.transform(labels)
        return (batch, labels)