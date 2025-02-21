import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from weather_app.data_and_model.model.utils import train_model, generator, scatter_data_sampling, count_r2, plot_loss, \
    PlotMode, test_model

np.set_printoptions(precision=8, suppress=True)
pd.set_option('display.float_format', '{:.10f}'.format)

print(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # gets last sequence
        return out


# read data
df_train = pd.read_csv('../data/history_data_small.csv')
df_train.set_index('time', inplace=True)

# data preparation
df_train_val = df_train.values
columns = df_train.columns
scaler = StandardScaler()
data = scaler.fit_transform(df_train_val)

# generators to create batches
lookback = 120
delay = 0
batch_size = 1024
train_gen = generator(data, lookback=lookback, delay=delay, min_index=0, max_index=round(0.7*len(data)),
                      batch_size=batch_size, device=device)
val_gen = generator(data, lookback=lookback, delay=delay, min_index=round(0.7*len(data))+1,
                     max_index=round(0.85*len(data)), batch_size=batch_size, device=device)
test_gen = generator(data, lookback=lookback, delay=delay, min_index=round(0.85*len(data))+1,
                    max_index=len(data)-1, batch_size=batch_size, device=device)


n_columns = 10
input_size = lookback * n_columns  # 120 * 10 = 1200
hidden_size = 1024
output_size = 10  # One row with 10 columns
epochs = 100

# define model
model = MyLSTM(n_columns, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses, val_losses, train_preds, train_targets, test_preds, test_targets = train_model(epochs, model, train_gen,
                                                                                             val_gen, criterion,
                                                                                             optimizer)

# rescale
train_predictions = scaler.inverse_transform(train_preds)
train_targets = scaler.inverse_transform(train_targets)
test_predictions = scaler.inverse_transform(test_preds)
test_targets = scaler.inverse_transform(test_targets)

# samples plots
scatter_data_sampling(test_targets, test_predictions, 0, 1, columns, PlotMode.Train)
scatter_data_sampling(train_targets, train_predictions, 0, 1, columns, PlotMode.Test)

# R2 score
count_r2(test_targets, test_predictions, columns)

# loss plot
plot_loss(epochs, train_losses, val_losses)

# test loss
test_model(model, test_gen, 50, criterion)

# more models - seqtoseq/ gru/ TCNs
# predicting sequences
# more refactoring, common stuff in LSTM and MLP
