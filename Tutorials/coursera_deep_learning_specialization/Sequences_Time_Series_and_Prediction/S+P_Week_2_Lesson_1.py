import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

print(15*'-')
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(10)
for x, y in dataset:
    print(x.numpy(), y.numpy())

print(15*'-')
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print(x.numpy(), y.numpy())
