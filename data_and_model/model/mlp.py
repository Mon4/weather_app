import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from weather_app.data_and_model.model.utils import count_r2, scatter_data_sampling, PlotMode, plot_loss, train_model, \
    generator

np.set_printoptions(precision=8, suppress=True)
pd.set_option('display.float_format', '{:.10f}'.format)


class MyModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear_inner = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_in(x)
        x = self.relu1(x)
        x = self.linear_inner(x)
        x = self.relu2(x)
        x = self.linear_output(x)
        return x


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
                      batch_size=batch_size)
val_gen = generator(data, lookback=lookback, delay=delay, min_index=round(0.7*len(data))+1,
                    max_index=round(0.85*len(data)), batch_size=batch_size)

# define model
n_columns = 10
input_size = lookback * n_columns  # 120 * 10 = 1200
hidden_size = 1024
output_size = 10  # One row with 10 columns
epochs = 100


# define model
model = MyModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train model
train_losses, val_losses, train_preds, train_targets, test_preds, test_targets = train_model(epochs, model, val_gen,
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
