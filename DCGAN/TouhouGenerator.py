import tensorflow as tf
import tensorflow.python.keras as keras

import glob
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras import layers
import time

class TouhouGenerator:
    def __init__(self, batchSize : int):
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = batchSize

        #self.train_dataset = tf.data.Dataset.from_tensor_slices(train_img).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)


    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16*16*512, use_bias=False, input_shape=(100,)))
        model.add(layers.Reshape((16, 16, 512)))

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        return model


    def make_discriminator_model(self):
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model