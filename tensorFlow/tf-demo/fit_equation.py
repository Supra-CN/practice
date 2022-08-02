# -*- coding: UTF-8 -*-
"""
一元一次函數中weights和biases的優化
"""
import sys
from tabulate import tabulate

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.autoscale(enable=True)
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  # plt.plot(hist['epoch'], hist['accuracy'], label = 'accuracy')
  # plt.ylim([0,5])
  plt.legend()

  # plt.figure()
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean Square Error [$MPG^2$]')
  # plt.plot(hist['epoch'], hist['mse'],
  #          label='Train Error')
  # plt.plot(hist['epoch'], hist['val_mse'],
  #          label = 'Val Error')
  # plt.ylim([0,20])
  # plt.legend()
  plt.show()

def transposing(data):
    return list(map(list, zip(*data)))


def transposed_table(data, headers):
    return tabulate(transposing(data), headers=headers)


def function(input, show=False):
    ret = input[0] ** input[1]
    if show:
        print(f' f({input}) = {ret}')

    return ret


def make_data(size, smp_max=sys.maxsize, smp_mix=-sys.maxsize - 1, show=False):
    if show:
        print(f'===')
        print(f'smp_max: {smp_max}')
        print(f'smp_mix: {smp_mix}')
    samples = np.random.rand(size, 2)
    tags = []

    for i in samples:
        tags.append(function(i, show))

    if (show):
        print(f'===')
        print(f'smp: {samples}')
        print(f'tag: {tags}')
    return [np.array(samples, dtype=float), np.array(tags, dtype=float)]


def run():
    function([8, 3], True)
    train_size = 100
    train_epochs = 100
    units_size = 1

    max = 10000

    data_train = make_data(train_size, smp_max=max, show=True)
    data_headers = ['sample', 'tag']
    print(f'data_train = \n{transposed_table(data_train, headers=data_headers)}')

    model = tf.keras.Sequential([
        # keras.layers.Flatten(input_shape=(28, 28)),
        # keras.layers.Dense(128),
        # tf.keras.layers.Dense(128, activation='relu', ),
        # tf.keras.layers.Dense(128, activation='relu', input_shape=[2]),
        tf.keras.layers.Dense(1,  input_shape=[2]),

        # tf.keras.layers.Dense(2),
        # tf.keras.layers.Dense(4),
        # tf.keras.layers.Dense(8),
        # tf.keras.layers.Dense(16),
        # tf.keras.layers.Dense(32),
        # tf.keras.layers.Dense(64),
        # tf.keras.layers.Dense(128),
        # tf.keras.layers.Dense(64),
        # tf.keras.layers.Dense(32),
        # tf.keras.layers.Dense(16),
        # tf.keras.layers.Dense(8),
        # tf.keras.layers.Dense(4),
        # tf.keras.layers.Dense(2),

        tf.keras.layers.Dense(units=units_size)
        # tf.keras.layers.Dense(units=units_size, input_shape=[2])
    ])

    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(0.1))
    history = model.fit(data_train[0], data_train[1], epochs=train_epochs, verbose=True)

    model.summary()

    plot_history(history)

    print(f'===')
    print(f'predict[8,3] = {model.predict([[8,3]])}')

    test_size = 100
    data_test = make_data(test_size, smp_max=max)

    model.evaluate(data_test[0], data_test[1])

    test_rets = model.predict(data_test[0])

    predicteds = []
    diffs = []
    for i in range(len(data_test[1])):
        predicted = test_rets[i][0]
        predicteds.append(predicted)
        diffs.append(predicted - data_test[1][i])

    data_test.append(predicteds)
    data_test.append(diffs)
    test_headers = ['test', 'tag', 'predicted', 'diff']
    print(f'data_test = \n{transposed_table(data_test, headers=test_headers)}')

    print(f'test_diff.mean = {np.mean(diffs)}')
    print(f'test_diff.average = {np.average(diffs)}')
    print(f'test_diff.var = {np.var(diffs)}')
    print(f'test_diff.std = {np.std(diffs)}')


def plot_predict():
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.autoscale(enable=True)
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  # plt.plot(hist['epoch'], hist['val_mae'],
  #          label = 'Val Error')
  # plt.ylim([0,5])
  plt.legend()

  # plt.figure()
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean Square Error [$MPG^2$]')
  # plt.plot(hist['epoch'], hist['mse'],
  #          label='Train Error')
  # plt.plot(hist['epoch'], hist['val_mse'],
  #          label = 'Val Error')
  # plt.ylim([0,20])
  # plt.legend()
  plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
