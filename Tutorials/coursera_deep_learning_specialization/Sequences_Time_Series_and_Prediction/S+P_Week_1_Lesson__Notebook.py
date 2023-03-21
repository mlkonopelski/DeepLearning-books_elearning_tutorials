# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, start=0, format='-', end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Series')
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def mse(true, pred):
    return ((true - pred) ** 2).mean()


def mae(true, pred):
    return np.abs(true - pred).mean()


# Prepare data
time = np.arange(4 * 365 + 1, dtype='float32')
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

## Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + noise(time, noise_level, seed=42)

# plt.figure(figsize=(10, 6))
# plot_series(time, series)
# plt.title('Full set')
# plt.show()

## Split series into train and validation
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

## Show train set
# plt.figure(figsize=(10, 6))
# plot_series(time_train, x_train)
# plt.title('x_train')
# plt.show()

## Show validation set
# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plt.title('x_valid')
# plt.show()

# Naive Forecast
naive_forecast = series[split_time-1:-1]

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid, end=150)
# plot_series(time_valid, naive_forecast, end=150)
# plt.title('Naive Forecast')
# plt.show()

print(mse(x_valid, naive_forecast))
print(mae(x_valid, naive_forecast))

# Moving Average
def moving_average(series, window_size, period_offset=0):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time+window_size].mean())
    return np.array(forecast)[split_time - period_offset - window_size:]

moving_avg = moving_average(series, 30)

# plt.plot(fig_size=(10, 6))
# plot_series(time_valid, x_valid, start=150)
# plot_series(time_valid, moving_avg, start=150)
# plt.title('moving average')
# plt.show()

print(mse(x_valid, moving_avg))
print(mae(x_valid, moving_avg))

# Diff Moving Average
diff_series = series[365:] - series[:-365]
diff_time = time[365:]

# plt.figure(figsize=(10, 6))
# plot_series(diff_time, diff_series)
# plt.title('Diff series')
# plt.show()

diff_moving_average = moving_average(diff_series, 50, 365)

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, diff_series[split_time-365:])
# plot_series(time_valid, diff_moving_average)
# plt.title('diff_movinng_average')
# plt.show()

diff_moving_avg_plus_past = series[split_time-365:-365] + diff_moving_average

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plot_series(time_valid, diff_moving_avg_plus_past)
# plt.title('diff moving average plus past')
# plt.show()

mse(x_valid, diff_moving_avg_plus_past)
mae(x_valid, diff_moving_avg_plus_past)

diff_moving_avg_plus_past_smooth = moving_average(series, 10) + diff_moving_average

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past_smooth)
plt.title('diff moving average plus past plus smooth')
plt.show()

mse(x_valid, diff_moving_avg_plus_past_smooth)
mae(x_valid, diff_moving_avg_plus_past_smooth)
