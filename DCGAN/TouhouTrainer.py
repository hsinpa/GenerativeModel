from DCGAN.TouhouGenerator import TouhouGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def sample_images(generator, z_dim, image_grid_rows=4, image_grid_columns=4):
    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

touhouGenerator = TouhouGenerator(5000)
generator = touhouGenerator.make_generator_model()

#noise = tf.random.normal([1, 100])
#sample_images(generator, 100)
# plt.imshow(generated_image[0, :, :, 0])
# plt.show()

