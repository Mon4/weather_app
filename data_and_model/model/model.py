import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim, tensor, float32


def generator(data, lookback, delay, min_index=0, max_index=None, batch_size=128):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if i + batch_size >= max_index:
            i = min_index + lookback
            # break

        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows), lookback, data.shape[-1]))
        targets = np.zeros((len(rows), 10))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j])
            samples[j] = data[indices]
            targets[j] = data[indices[-1] + 1 + delay]

        yield samples, targets


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear_inner = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_in(x)
        x = self.relu1(x)
        x = self.linear_inner(x)
        x = self.relu2(x)
        x = self.sigmoid(x)
        return x


def scatter_data_sampling(targets_x, targets_y, predictions_x, predictions_y):
    plt.scatter(targets_x, targets_y, label='targets')
    plt.scatter(predictions_x, predictions_y, label='predictions', alpha=0.3)
    plt.xlabel('temp')
    plt.ylabel('wind')
    plt.legend()
    plt.show()


def plot_loss(epochs: int, train_loss: list, val_loss: list) -> None:
    plt.plot(range(1, epochs+1), train_loss, label='train')
    plt.plot(range(1, epochs+1), val_loss, label='validation')
    plt.title("Loss function")
    plt.xlabel("Epochs")
    plt.ylabel("Loss values")
    plt.show()


def train_model(epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        x_batch, y_batch = next(train_gen)  # Get batch from generator
        x_val, y_val = next(val_gen)

        x_batch = tensor(x_batch, dtype=float32)
        y_batch = tensor(y_batch, dtype=float32)

        x_val = tensor(x_val, dtype=float32)
        y_val = tensor(y_val, dtype=float32)

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        train_losses.append(loss.item())

        # Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_predictions.extend(outputs.detach().numpy())
        train_targets.extend(y_batch.numpy())

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # evaluation
        model.eval()  # set model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            outputs = model(x_val)  # Forward pass
            val_loss = criterion(outputs, y_val).item()  # Calculate loss
            val_losses.append(val_loss)

            test_predictions.extend(outputs.detach().numpy())
            test_targets.extend(y_batch.numpy())
            print(f'Validation loss: {val_loss}')

        # scatter_data_sampling(y_batch[:, 0], y_batch[:, 1], outputs[:, 0], outputs[:, 1])
    return train_losses, val_losses


# read data
df_train = pd.read_csv('../data/history_data_small.csv')
df_train.set_index('time', inplace=True)

# data preparation
df_train_val = df_train.values
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df_train_val)

# generator to create batches
lookback = 120
delay = 0
batch_size = 128
train_gen = generator(data, lookback=lookback, delay=delay, min_index=0, max_index=round(0.7*len(data)),
                      batch_size=batch_size)
val_gen = generator(data, lookback=lookback, delay=delay, min_index=round(0.7*len(data))+1,
                    max_index=round(0.85*len(data)), batch_size=batch_size)

# define model
n_columns = 10
input_size = lookback * n_columns  # 120 * 10 = 1200
hidden_size = 1024
output_size = 10  # One row with 10 columns
epochs = 200

train_predictions = []
train_targets = []
test_predictions = []
test_targets = []

# define model
model = MyModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
train_losses, val_losses = train_model(epochs)


# rescale
train_predictions = scaler.inverse_transform(train_predictions)
train_targets = scaler.inverse_transform(train_targets)
test_predictions = scaler.inverse_transform(test_predictions)
test_targets = scaler.inverse_transform(test_targets)

# plot test
temp_test_pred = [pred[0] for pred in test_predictions]
temp_test_target = [target[0] for target in test_targets]

wind_test_pred = [pred[1] for pred in test_predictions]
wind_test_target = [target[1] for target in test_targets]

scatter_data_sampling(temp_test_target, wind_test_target, temp_test_pred, wind_test_pred)  # to do, better object oriented this func

# plot train
temp_train_pred = [pred[0] for pred in train_predictions]
temp_train_target = [target[0] for target in test_targets]

wind_train_pred = [pred[1] for pred in train_predictions]
wind_train_target = [target[1] for target in test_targets]

scatter_data_sampling(temp_train_target, wind_train_target, temp_train_pred, wind_train_pred)  # to do, better object oriented this func


# R score
r2 = r2_score(y_true=temp_test_target, y_pred=temp_test_pred)
print(f'R² Score: {r2}')

plot_loss(epochs, train_losses, val_losses)

# why so bad wind prediction?
