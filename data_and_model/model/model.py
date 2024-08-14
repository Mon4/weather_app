import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('../data/history_data_small.csv')
df_train.set_index('time', inplace=True)

data_train = df_train.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(data_train)


def create_sequences(data, loopback, delay):
    X = []
    y = []
    for i in range(loopback, len(data)):
        X.extend(data[i-loopback:i])
        y.extend(data[i:i+delay])
    print('x')
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


loopback = 120
delay = 72
X_train, y_train = create_sequences(scaled_data_train, loopback, delay)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

