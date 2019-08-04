from DCGAN.TouhouGenerator import TouhouGenerator
from DataLoader.ImageUtility import ImageUtility
from DataLoader.LoaderUtility import LoaderUtility
import numpy as np
import matplotlib.pyplot as plt

resizeFolder = "../Dataset/TouhouDataset/ResizeFolder/"
valid_images = [".jpg",".png", ".jpeg"]

img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)

z_dim = 100

iterations = 200
batch_size = 16
sample_interval = 10

imageLoader = ImageUtility()
loaderUtility = LoaderUtility()
touhouGenerator = TouhouGenerator(5000, img_shape, z_dim)

#noise = tf.random.normal([1, 100])
#sample_images(generator, 100)
# plt.imshow(generated_image[0, :, :, 0])
# plt.show()

touhouImageSet = loaderUtility.GetDatasetFromPath(resizeFolder, valid_images, [], loaderUtility.TanhNormalized)[0]

touhouGenerator.train(touhouImageSet, iterations, batch_size, sample_interval)

imageLoader.sample_images(touhouGenerator.generator, z_dim)
