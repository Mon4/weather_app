import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generator(data, lookback, delay, min_index=0, max_index=None, batch_size=128):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if i + batch_size >= max_index:
            i = min_index + lookback

        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows), lookback, data.shape[-1]))
        targets = np.zeros((len(rows), 10))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j])
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay, 0]

        yield samples, targets


df_train = pd.read_csv('../data/history_data_small.csv')
df_train.set_index('time', inplace=True)

data_train = df_train.values

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data_train)


lookback = 120
delay = 72
batch_size = 128
train_gen = generator(data, lookback=lookback, delay=delay, min_index=0, max_index=round(0.7*len(data)),
                      batch_size=batch_size)
val_gen = generator(data, lookback=lookback, delay=delay, min_index=round(0.7*len(data))+1,
                    max_index=round(0.85*len(data)), batch_size=batch_size)

for s, t in train_gen:
    print(s.shape, t.shape)
    