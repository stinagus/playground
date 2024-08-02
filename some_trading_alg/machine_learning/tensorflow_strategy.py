from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random


def download_data(symbol, start_date, end_date):
    raw = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
    data = pd.DataFrame(raw)
    data['return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['direction'] = np.where(data['return'] > 0, 1, 0)
    return data.dropna()


def add_lags(data, lags=5):
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['return'].shift(lag)
        cols.append(col)
    return data.dropna(), cols


def normalize_data(train_set, test_set):
    mu, std = train_set.mean(), train_set.std()
    training_data = (train_set - mu) / std
    test_data = (test_set - mu) / std
    return training_data, test_data, mu, std


def build_model(input_shape, optimizer):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels, epochs=50, validation_split=0.2):
    history = model.fit(train_data, train_labels,
                        epochs=epochs, verbose=False,
                        validation_split=validation_split,
                        shuffle=False)
    return history


def calculate_strategy_performance(data, predictions):
    data['prediction'] = np.where(predictions > 0, 1, -1)
    data['strategy'] = (data['prediction'] * data['return'])
    return data[['return', 'strategy']].sum().apply(np.exp), data[['return', 'strategy']].cumsum().apply(np.exp)


def plot_results(history, test_data, predictions):
    res = pd.DataFrame(history.history)
    res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
    plt.show()

    test_data['prediction'] = np.where(predictions > 0, 1, -1) # Make into short/long positions
    test_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()


def add_additional_features(data, cols):
    data['momentum'] = data['return'].rolling(5).mean().shift(1)
    data['volatility'] = data['return'].rolling(20).std().shift(1)
    data['distance'] = (data['Adj Close'] - data['Adj Close'].rolling(50).mean()).shift(1)
    cols.extend(['momentum', 'volatility', 'distance'])
    return data.dropna(), cols


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    ticker_symbol = "^GSPC"  # SP500
    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    lags = 5
    optimizer = Adam(learning_rate=0.0001)

    data = download_data(symbol=ticker_symbol, start_date=start_date, end_date=end_date)
    data, lag_cols = add_lags(data=data, lags=lags)
    data, cols = add_additional_features(data=data, cols=lag_cols)

    train_set, test_set = np.split(data, [int(.80 * len(data))])
    training_data, test_data, mu, std = normalize_data(train_set=train_set, test_set=test_set)

    set_seeds()
    model = build_model(input_shape=(len(cols),), optimizer=optimizer)
    history = train_model(model=model, train_data=training_data[cols], train_labels=train_set['direction'])

    model.evaluate(training_data[cols], train_set['direction'])

    train_predictions = np.where(model.predict(training_data[cols]) > 0.5, 1, 0) # Check confidence in prediction
    train_results = calculate_strategy_performance(data=train_set, predictions=train_predictions)

    plot_results(history=history, test_data=train_set, predictions=train_predictions)

    model.evaluate(test_data[cols], test_set['direction'])

    test_predictions = np.where(model.predict(test_data[cols]) > 0.5, 1, 0)
    test_results = calculate_strategy_performance(data=test_set, predictions=test_predictions)

    plot_results(history=history, test_data=test_set, predictions=test_predictions)
