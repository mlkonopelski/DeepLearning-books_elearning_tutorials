import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)


def trend(time, slope=0):
    return time * slope


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=0, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# create time and series
time = np.arange(4 * 365 + 1, dtype='float32')
series = 10 + trend(time, 0.05) + seasonality(time, 365, 40) + noise(time, 5, 42)

# create sets
split_time = 1000
time_train = time[:split_time]
series_train = series[:split_time]
time_valid = time[split_time:]
series_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def windowed_datasets(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


tf.random.set_seed(51)
np.random.seed(51)

# Model with variable LR

train_set = windowed_datasets(series_train, window_size, batch_size=128, shuffle_buffer_size=shuffle_buffer_size)

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)

model.compile(optimizer=tf.keras.optimizers.SGD(1e-8, momentum=0.9),
              loss=tf.keras.losses.Huber(),
              metrics=['mae'])

history = model.fit(train_set,
                    epochs=100,
                    callbacks=[lr_schedule])

plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([1e-8, 1e-4, 0, 30])
#plt.show()

# Model with new LR
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100.0)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=5e-5, momentum=0.9),
              loss=tf.keras.losses.Huber(),
              metrics=['mae'])
model.fit(train_set,
          epochs=400)


# Predict new data
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=[10, 6])
plot_series(time_valid, series_valid)
plot_series(time_valid, results)
plt.show()

print(tf.metrics.mae(series_valid, results).numpy())
