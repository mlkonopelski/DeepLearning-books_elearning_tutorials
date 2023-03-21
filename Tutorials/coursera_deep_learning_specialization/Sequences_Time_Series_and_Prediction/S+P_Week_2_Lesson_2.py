import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 354 + 1, dtype='float32')
baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + \
         trend(time, slope) + \
         seasonality(time, period=365, amplitude=amplitude) + \
         white_noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(1)
    return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.Sequential([l0])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), loss=tf.keras.metrics.mse, metrics=['mse'])
model.fit(dataset, epochs=100, verbose=0)

def val_model(model):
    forecast = []

    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time - window_size:]
    results = np.array(forecast)[:, 0, 0]

    return results

results = val_model(model)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(f'OneLayer DNN mse: {tf.keras.metrics.mae(x_valid, results).numpy()}')

l0 = keras.layers.Dense(10, activation='relu', input_shape=[window_size])
l1 = keras.layers.Dense(10, activation='relu')
output = keras.layers.Dense(1)
model = keras.models.Sequential([l0, l1, output])
model.compile(optimizer=keras.optimizers.SGD(1e-6, momentum=0.9), loss='mse')
model.fit(dataset, epochs=100, verbose=0)

results = val_model(model)

print(f'3 layers DNN mse: {tf.keras.metrics.mae(x_valid, results).numpy()}')


l0 = keras.layers.Dense(10, activation='relu', input_shape=[window_size])
l1 = keras.layers.Dense(10, activation='relu')
output = keras.layers.Dense(1)
model = keras.models.Sequential([l0, l1, output])

model.compile(optimizer=keras.optimizers.SGD(1e-8, momentum=0.9), loss='mse')
lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)

results = val_model(model)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(f'3 layers DNN mse with lr schedule: {tf.keras.metrics.mae(x_valid, results).numpy()}')
