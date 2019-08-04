import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import os
from tensorflow.python.keras import layers
import time
from tensorflow.python.keras.optimizers import Adam

class TouhouGenerator:
    def __init__(self, batchSize : int, img_shape, z_dim : int):
        self.BUFFER_SIZE = 6000
        self.BATCH_SIZE = batchSize
        self.img_shape = img_shape
        self.z_dim = z_dim

        self.discriminator = self.make_discriminator_model(img_shape)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        self.generator = self.make_generator_model(z_dim)
        self.discriminator.trainable = False
        self.gan = self.build_gan(self.generator, self.discriminator)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())

    def make_generator_model(self, z_dim):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(z_dim,)))
        model.add(layers.Reshape((16, 16, 256)))

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

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        return model


    def make_discriminator_model(self, img_shape):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def build_gan(self, generator, discriminator):
        model = tf.keras.Sequential()

        model.add(generator)
        model.add(discriminator)

        return model

    def train(self, raw_image, iterations, batch_size, sample_interval):
        losses = []
        accuracies = []
        iteration_checkpoints = []

        #raw_image = np.expand_dims(raw_image, axis=3)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            idx = np.random.randint(0, raw_image.shape[0], batch_size)
            imgs = raw_image[idx]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z)
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z)
            g_loss = self.gan.train_on_batch(z, real)

            if (iteration + 1) % sample_interval == 0:
                losses.append((d_loss, g_loss))

                accuracies.append(100.0 * accuracy)
                iteration_checkpoints.append(iteration + 1)
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (iteration + 1, d_loss, 100.0 * accuracy, g_loss))