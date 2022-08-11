# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tf_gpu_config
import random
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

def tf_demo_fashion_mnist():
    print(f'tf version: {tf.__version__}')

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f'train image  shape: {train_images.shape}')
    print(f'train label length: {len(train_labels)}')
    print(f'train labels: {train_labels}')
    print(f'test image labels: {test_images.shape}')
    print(f'test label length: {len(test_labels)}')
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # print('plt.show()')
    # plt.show()
    # print('plt.show() over')

    # print(f'train_images[0] b4: {train_images[0]}')

    train_images = train_images / 255.0
    # print(f'train_images[0] af: {train_images[0]}')

    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.colorbar()
        plt.xlabel(class_names[train_labels[i]])

    print('plt.show()')
    plt.show()
    print('plt.show() over')

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    print(f'predictions[0]： {predictions[0]}')
    print(f'predictions[0] max： {np.argmax(predictions[0])}')
    print(f'predictions[0]  is： {test_labels[0]}')

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    index = 0
    max_index = 100
    while index < max_index and index < len(test_labels):
        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 5
        num_cols = 4
        num_images = num_rows * num_cols
        # index += num_images
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(index, predictions[index], test_labels, test_images, class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(index, predictions[index], test_labels)
            index += 1
        plt.tight_layout()
        plt.show()

    # Grab an image from the test dataset.
    img = test_images[1]

    print(f'img.shape sample: {img.shape}')
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img,0))

    print(f'img.shape array : {img.shape}')

    predictions_single = probability_model.predict(img)

    print(f'predictions_single: {predictions_single}')

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)

    print(f'{np.argmax(predictions_single[0])}')


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def tf_demo_try():

    train_smps = []
    train_tags = []
    train_range = 3
    smp_max = sys.maxsize
    smp_mix = -sys.maxsize - 1
    print(f'===')
    print(f'smp_max: {smp_max}')
    print(f'smp_mix: {smp_mix}')

    for i in range(train_range):
        sample = random.randint(smp_mix, smp_max)
        train_smps.append(sample)
        train_tags.append(round(sample / 2))

    print(f'===')
    print(f'smp: {train_smps}')
    print(f'tag: {train_tags}')

    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512),
        keras.layers.Dense(128),
        # keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # model.fit(x_train, y_train, epochs=5)
    #
    # model.evaluate(x_test, y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # with tf.device("/cpu:0"):
    tf_demo_mnist()
    # tf_demo_fashion_mnist()
        # tf_demo_try()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
