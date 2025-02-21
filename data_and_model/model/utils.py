from enum import Enum
from typing import Generator

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch import tensor, float32, nn


class PlotMode(Enum):
    Train = 0
    Test = 1
    Validation = 2


def count_r2(test_targets: np.ndarray, test_predictions: np.ndarray, columns: list) -> None:
    print(f'R² Score:')
    for i in range(0, len(columns)):
        r2 = r2_score(y_true=test_targets[:, i], y_pred=test_predictions[:, i])
        print(f'{columns[i]}: {r2}')


def scatter_data_sampling(targets, predictions, x_id, y_id, columns: list, flag: PlotMode = PlotMode.Train) -> None:
    plt.scatter(targets[:, x_id], targets[:, y_id], label='targets')
    plt.scatter(predictions[:, x_id], predictions[:, y_id], label='predictions', alpha=0.3)
    plt.xlabel(columns[x_id])
    plt.ylabel(columns[y_id])
    plt.title(flag.name)
    plt.legend()
    plt.show()


def plot_loss(epochs: int, train_loss: list, val_loss: list) -> None:
    plt.plot(range(1, epochs + 1), train_loss, label='train')
    plt.plot(range(1, epochs + 1), val_loss, label='validation')
    plt.title("Loss function")
    plt.xlabel("Epochs")
    plt.ylabel("Loss values")
    plt.legend()
    plt.show()


def generator(data: np.ndarray, lookback: int, delay: int, device: torch.device,
              min_index: int = 0, max_index: int = None, batch_size: int = 128, ) -> (list, list):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if i + batch_size >= max_index:
            i = min_index + lookback
            # break

        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = torch.zeros((len(rows), lookback, data.shape[-1]))
        targets = torch.zeros((len(rows), 10))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j])
            samples[j] = tensor(data[indices], dtype=float32)
            targets[j] = tensor(data[indices[-1] + 1 + delay], dtype=float32)

        yield samples.to(device), targets.to(device)


def train_model(epochs: int, model: nn.Module, train_gen: Generator, val_gen: Generator, criterion: nn, optimizer: torch.optim) -> (list, ...):
    train_losses = []
    val_losses = []

    train_predictions = []
    train_targets = []

    test_predictions = []
    test_targets = []

    for epoch in range(epochs):
        model.train()

        # get batch from generator
        x_batch, y_batch = next(train_gen)
        x_val, y_val = next(val_gen)

        # forward pass
        train_outputs = model(x_batch)
        loss = criterion(train_outputs, y_batch)
        train_losses.append(loss.item())

        # backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_predictions.extend(train_outputs.cpu().detach().numpy())
        train_targets.extend(y_batch.cpu().numpy())

        # evaluation
        model.eval()

        with torch.no_grad():  # disable gradient computation
            test_outputs = model(x_val)  # forward pass
            val_loss = criterion(test_outputs, y_val).item()
            val_losses.append(val_loss)

            test_predictions.extend(test_outputs.cpu().detach().numpy())
            test_targets.extend(y_val.cpu().numpy())

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss} ')

    return train_losses, val_losses, train_predictions, train_targets, test_predictions, test_targets


def test_model(model: nn.Module, test_gen: Generator, steps: int, criterion: nn) -> None:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for step in range(steps):
            test_inputs, test_labels = next(test_gen)
            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs.squeeze(), test_labels).item()
    print(f'Test Loss: {test_loss/steps:.4f}')
